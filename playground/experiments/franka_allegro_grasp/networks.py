"""Neural network architectures for DEXTRAH-G teacher-student training.

Teacher: LSTM(1024) actor + LSTM(2048) critic (asymmetric, central value)
Student: CNN depth encoder + LSTM(512) + MLP + auxiliary object position head

Reference:
- DEXTRAH-G: dextrah_lab/distillation/a2c_with_aux_depth.py
- rl_games configs: rl_games_ppo_lstm_cfg.yaml, rl_games_ppo_lstm_scratch_cnn_aux.yaml
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
    """Initialize layer with orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RunningMeanStd:
    """Scalar running mean and standard deviation tracker for value normalization."""

    def __init__(self, shape: tuple = (), device: str = "cuda", epsilon: float = 1e-5):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var) + self.mean


class ObsRunningMeanStd:
    """Per-feature running mean and std for observation normalization (rl_games style).

    Unlike scalar RunningMeanStd (for value normalization), this maintains
    separate statistics for each observation feature dimension.
    """

    def __init__(self, shape: tuple, device: str = "cuda", epsilon: float = 1e-5):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        """Update statistics with new observations.

        Args:
            x: Observations (..., features). All dims except last are flattened.
        """
        x_flat = x.reshape(-1, x.shape[-1])
        batch_mean = x_flat.mean(dim=0)
        batch_var = x_flat.var(dim=0, unbiased=False)
        batch_count = x_flat.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        """Normalize and clip observations."""
        normalized = (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        return torch.clamp(normalized, -clip, clip)


def _build_mlp(input_dim: int, hidden_dims: tuple, activation: str = "elu") -> nn.Sequential:
    """Build MLP with given hidden dims and activation."""
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}.get(activation, nn.ELU)
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(layer_init(nn.Linear(prev_dim, dim)))
        layers.append(act_fn())
        prev_dim = dim
    return nn.Sequential(*layers)


class DepthEncoder(nn.Module):
    """CNN depth encoder (DEXTRAH-G CustomCNN).

    Architecture:
        Conv2d(1, 16, k=6, s=2) -> ReLU -> LayerNorm
        Conv2d(16, 32, k=4, s=2) -> ReLU -> LayerNorm
        Conv2d(32, 64, k=4, s=2) -> ReLU -> LayerNorm
        Conv2d(64, 128, k=4, s=2) -> ReLU -> LayerNorm
        AdaptiveAvgPool2d(1,1) -> Linear(128, output_dim)
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: tuple = (16, 32, 64, 128),
        kernel_sizes: tuple = (6, 4, 4, 4),
        strides: tuple = (2, 2, 2, 2),
        output_dim: int = 128,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        conv_layers = []
        in_ch = input_channels
        for i, (out_ch, ks, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride))
            conv_layers.append(nn.ReLU())
            if use_layer_norm:
                # LayerNorm is applied after computing spatial dims dynamically
                conv_layers.append(_DeferredLayerNorm(out_ch))
            in_ch = out_ch

        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode depth image.

        Args:
            x: Depth image (batch, 1, H, W)

        Returns:
            features: Encoded features (batch, output_dim)
        """
        features = self.conv(x)
        features = features.view(features.shape[0], -1)
        return self.fc(features)


class _DeferredLayerNorm(nn.Module):
    """LayerNorm that defers shape computation to first forward pass."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            # x shape: (B, C, H, W)
            self.norm = nn.LayerNorm([self.num_channels, x.shape[2], x.shape[3]], device=x.device)
        return self.norm(x)


class LSTMBlock(nn.Module):
    """LSTM block with optional LayerNorm (rl_games style).

    LSTM output is LayerNorm-ed before passing to MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, use_layer_norm: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through LSTM.

        Args:
            x: Input tensor. Either (batch, features) for single step
               or (batch, seq_len, features) for sequence.
            hidden: (h_0, c_0) each of shape (num_layers, batch, hidden_dim).
                    If None, zeros are used.
            dones: Done flags (batch,) or (batch, seq_len) for resetting hidden.

        Returns:
            output: LSTM output with LayerNorm applied.
            new_hidden: Updated (h_n, c_n).
        """
        batch_size = x.shape[0]
        is_sequence = x.dim() == 3

        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h_0, c_0)

        if is_sequence:
            # Sequence mode: process entire sequence
            # Reset hidden states at episode boundaries
            if dones is not None:
                outputs = []
                h, c = hidden
                for t in range(x.shape[1]):
                    # Reset hidden where episodes ended
                    if t > 0:
                        reset_mask = dones[:, t - 1].unsqueeze(0).unsqueeze(-1)
                        h = h * (1.0 - reset_mask)
                        c = c * (1.0 - reset_mask)
                    out, (h, c) = self.lstm(x[:, t : t + 1, :], (h, c))
                    outputs.append(out)
                output = torch.cat(outputs, dim=1)
                new_hidden = (h, c)
            else:
                output, new_hidden = self.lstm(x, hidden)
            # Apply layer norm
            output = self.layer_norm(output)
        else:
            # Single step mode
            x_seq = x.unsqueeze(1)  # (batch, 1, features)
            output, new_hidden = self.lstm(x_seq, hidden)
            output = self.layer_norm(output).squeeze(1)  # (batch, hidden_dim)

        return output, new_hidden

    def init_hidden(self, batch_size: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """Create initial hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


class TeacherActorCritic(nn.Module):
    """Teacher actor-critic with asymmetric LSTM networks (rl_games style).

    Actor (before_mlp=True, concat_input=True):
        obs -> LSTM(1024) -> LayerNorm -> [concat obs] -> MLP[512,512] -> action(11)
    Critic (before_mlp=False, concat_input=True):
        obs -> MLP[1024,512] -> [concat obs] -> LSTM(2048) -> LayerNorm -> value(1)

    The actor and critic are fully separate networks (asymmetric central value).
    Skip connections (concat_input) are critical for LSTM policy training.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_lstm_units: int = 1024,
        critic_lstm_units: int = 2048,
        actor_mlp_dims: tuple = (512, 512),
        critic_mlp_dims: tuple = (1024, 512),
        lstm_layers: int = 1,
        lstm_layer_norm: bool = True,
        activation: str = "elu",
        init_sigma: float = 0.0,
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        # Actor (rl_games before_mlp=True, concat_input=True):
        # obs -> LSTM -> [concat LSTM_out, obs] -> MLP -> action
        self.actor_lstm = LSTMBlock(num_actor_obs, actor_lstm_units, lstm_layers, lstm_layer_norm)
        self.actor_mlp = _build_mlp(actor_lstm_units + num_actor_obs, actor_mlp_dims, activation)
        self.actor_mean = layer_init(nn.Linear(actor_mlp_dims[-1], num_actions), std=0.01)
        self.actor_log_std = nn.Parameter(torch.full((num_actions,), init_sigma))

        # Critic (rl_games before_mlp=False, concat_input=True):
        # obs -> MLP -> [concat MLP_out, obs] -> LSTM -> value
        self.critic_mlp = _build_mlp(num_critic_obs, critic_mlp_dims, activation)
        self.critic_lstm = LSTMBlock(
            critic_mlp_dims[-1] + num_critic_obs, critic_lstm_units, lstm_layers, lstm_layer_norm
        )
        self.critic_head = layer_init(nn.Linear(critic_lstm_units, 1))

    def forward_actor(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Actor forward pass (before_mlp=True, concat_input=True).

        Flow: obs -> LSTM -> [concat LSTM_out, obs] -> MLP -> action

        Returns:
            action_mean, action_std, new_hidden
        """
        lstm_out, new_hidden = self.actor_lstm(obs, hidden, dones)
        # Skip connection: concat LSTM output with original input
        features = torch.cat([lstm_out, obs], dim=-1)
        features = self.actor_mlp(features)
        action_mean = self.actor_mean(features)

        log_std = torch.clamp(self.actor_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(log_std)

        return action_mean, action_std, new_hidden

    def forward_critic(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Critic forward pass (before_mlp=False, concat_input=True).

        Flow: obs -> MLP -> [concat MLP_out, obs] -> LSTM -> value

        Returns:
            value, new_hidden
        """
        mlp_out = self.critic_mlp(obs)
        # Skip connection: concat MLP output with original input
        lstm_input = torch.cat([mlp_out, obs], dim=-1)
        lstm_out, new_hidden = self.critic_lstm(lstm_input, hidden, dones)
        value = self.critic_head(lstm_out).squeeze(-1)
        return value, new_hidden

    def get_action_and_value(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actor_hidden: tuple | None = None,
        critic_hidden: tuple | None = None,
        action: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict:
        """Get action, log_prob, entropy, value, and new hidden states."""
        # Sanitize inputs
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        actor_obs = torch.clamp(actor_obs, -100.0, 100.0)
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        critic_obs = torch.clamp(critic_obs, -100.0, 100.0)

        # Actor
        action_mean, action_std, new_actor_hidden = self.forward_actor(actor_obs, actor_hidden, dones)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        action_std = torch.nan_to_num(action_std, nan=1.0, posinf=1.0, neginf=0.01)
        action_std = torch.clamp(action_std, min=1e-6)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)

        # Critic
        value, new_critic_hidden = self.forward_critic(critic_obs, critic_hidden, dones)

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "action_mean": action_mean,
            "action_std": action_std,
            "actor_hidden": new_actor_hidden,
            "critic_hidden": new_critic_hidden,
        }

    def get_value(
        self,
        critic_obs: torch.Tensor,
        critic_hidden: tuple | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Get value estimate only."""
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        critic_obs = torch.clamp(critic_obs, -100.0, 100.0)
        return self.forward_critic(critic_obs, critic_hidden, dones)

    def init_hidden(self, batch_size: int, device: str = "cuda"):
        """Initialize hidden states for both actor and critic."""
        return {
            "actor": self.actor_lstm.init_hidden(batch_size, device),
            "critic": self.critic_lstm.init_hidden(batch_size, device),
        }


class StudentNetwork(nn.Module):
    """Student network with CNN depth encoder + LSTM + MLP + auxiliary head.

    Architecture:
        Depth image -> CNN -> 128D features
        Concat(CNN_out, proprio_obs) -> LSTM(512) -> MLP[512,512,256] -> action(11)
        LSTM output -> AuxMLP[512,256] -> object_pos(3)

    Reference: DEXTRAH-G a2c_with_aux_cnn.py
    """

    def __init__(
        self,
        num_proprio_obs: int,
        num_actions: int,
        depth_channels: tuple = (16, 32, 64, 128),
        depth_kernel_sizes: tuple = (6, 4, 4, 4),
        depth_strides: tuple = (2, 2, 2, 2),
        cnn_output_dim: int = 128,
        cnn_use_layer_norm: bool = True,
        lstm_units: int = 512,
        lstm_layers: int = 1,
        lstm_layer_norm: bool = True,
        mlp_dims: tuple = (512, 512, 256),
        aux_mlp_dims: tuple = (512, 256),
        aux_output_dim: int = 3,
        activation: str = "elu",
        init_sigma: float = 0.0,
    ):
        super().__init__()
        self.num_proprio_obs = num_proprio_obs
        self.num_actions = num_actions

        # CNN depth encoder
        self.depth_encoder = DepthEncoder(
            input_channels=1,
            channels=depth_channels,
            kernel_sizes=depth_kernel_sizes,
            strides=depth_strides,
            output_dim=cnn_output_dim,
            use_layer_norm=cnn_use_layer_norm,
        )

        # LSTM: takes CNN features + proprio
        lstm_input_dim = cnn_output_dim + num_proprio_obs
        self.lstm = LSTMBlock(lstm_input_dim, lstm_units, lstm_layers, lstm_layer_norm)

        # Policy MLP
        self.policy_mlp = _build_mlp(lstm_units, mlp_dims, activation)
        self.policy_mean = layer_init(nn.Linear(mlp_dims[-1], num_actions), std=0.01)
        self.policy_log_std = nn.Parameter(torch.full((num_actions,), init_sigma))

        # Auxiliary head (object position prediction)
        self.aux_mlp = _build_mlp(lstm_units, aux_mlp_dims, activation)
        self.aux_head = layer_init(nn.Linear(aux_mlp_dims[-1], aux_output_dim))

        # Value head
        self.value_mlp = _build_mlp(lstm_units, mlp_dims, activation)
        self.value_head = layer_init(nn.Linear(mlp_dims[-1], 1))

    def forward(
        self,
        depth_image: torch.Tensor,
        proprio_obs: torch.Tensor,
        hidden: tuple | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            depth_image: (batch, 1, H, W) or (batch, seq, 1, H, W) depth images
            proprio_obs: (batch, proprio_dim) or (batch, seq, proprio_dim) proprioception
            hidden: LSTM hidden state
            dones: Done flags for LSTM reset

        Returns:
            Dictionary with action_mean, action_std, value, aux_pred, hidden
        """
        is_sequence = proprio_obs.dim() == 3

        if is_sequence:
            batch, seq_len = proprio_obs.shape[:2]
            # Encode each depth frame through CNN
            depth_flat = depth_image.reshape(batch * seq_len, *depth_image.shape[2:])
            cnn_features = self.depth_encoder(depth_flat)
            cnn_features = cnn_features.reshape(batch, seq_len, -1)
            # Concat with proprio
            lstm_input = torch.cat([cnn_features, proprio_obs], dim=-1)
        else:
            # Single step
            cnn_features = self.depth_encoder(depth_image)
            lstm_input = torch.cat([cnn_features, proprio_obs], dim=-1)

        # LSTM
        lstm_out, new_hidden = self.lstm(lstm_input, hidden, dones)

        # Policy
        policy_features = self.policy_mlp(lstm_out)
        action_mean = self.policy_mean(policy_features)
        log_std = torch.clamp(self.policy_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(log_std)

        # Value
        value_features = self.value_mlp(lstm_out)
        value = self.value_head(value_features).squeeze(-1)

        # Auxiliary prediction
        aux_features = self.aux_mlp(lstm_out)
        aux_pred = self.aux_head(aux_features)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "aux_pred": aux_pred,
            "hidden": new_hidden,
        }

    def get_action_and_value(
        self,
        depth_image: torch.Tensor,
        proprio_obs: torch.Tensor,
        hidden: tuple | None = None,
        action: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict:
        """Get action, log_prob, entropy, value, aux_pred."""
        out = self.forward(depth_image, proprio_obs, hidden, dones)

        dist = Normal(out["action_mean"], out["action_std"])
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": out["value"],
            "action_mean": out["action_mean"],
            "action_std": out["action_std"],
            "aux_pred": out["aux_pred"],
            "hidden": out["hidden"],
        }

    def init_hidden(self, batch_size: int, device: str = "cuda"):
        return self.lstm.init_hidden(batch_size, device)


def gaussian_kl(
    mu0: torch.Tensor, sigma0: torch.Tensor, mu1: torch.Tensor, sigma1: torch.Tensor
) -> torch.Tensor:
    """Analytical KL divergence between two diagonal Gaussians (rl_games style)."""
    c1 = torch.log(sigma1 / (sigma0 + 1e-5) + 1e-5)
    c2 = (sigma0**2 + (mu0 - mu1) ** 2) / (2.0 * (sigma1**2 + 1e-5))
    c3 = -0.5
    kl = c1 + c2 + c3
    return kl.sum(dim=-1).mean()
