"""Random Network Adversary (RNA) for sim-to-real robustness.

Reference: OpenAI et al. 2019 (https://arxiv.org/abs/1910.07113) Section B.3
Original: IsaacGymEnvs/isaacgymenvs/utils/rna_util.py

RNA adds random perturbations to actions to account for unmodelled dynamics.
Uses a random neural network with per-env dropout masks, outputting softmax
distributions over discretised action bins.

Key design choices from the paper:
- Softmax over discrete bins (not continuous tanh) to avoid actions near 0
- Dropout masks approximate per-env diverse networks on GPU
- Weights re-initialised periodically for changing perturbation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomNetworkAdversary(nn.Module):
    """Random network that generates adversarial action perturbations.

    Architecture:
        input (in_dims) → Linear(512) → ReLU → Linear(512) → dropout1
        → Linear(1024) → ReLU → Linear(1024) → dropout2
        → Linear(out_dims * softmax_bins) → reshape → softmax

    Dropout masks are sampled per-env to approximate having different
    random networks for each environment (as in the CPU-based OpenAI
    implementation) while running efficiently on GPU.

    Args:
        num_envs: Number of parallel environments
        in_dims: Input dimensions (dof_pos + object_pose = 16 + 7 = 23)
        out_dims: Output dimensions (number of action dims = 16)
        softmax_bins: Number of discretisation bins per output dim (32)
        device: Torch device
    """

    def __init__(
        self,
        num_envs: int,
        in_dims: int,
        out_dims: int,
        softmax_bins: int = 32,
        device: str = "cuda",
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.softmax_bins = softmax_bins
        self.num_envs = num_envs
        self.device = device

        self.num_feats1 = 512
        self.num_feats2 = 1024

        # Network layers
        self.fc1 = nn.Linear(in_dims, self.num_feats1)
        self.fc1_1 = nn.Linear(self.num_feats1, self.num_feats1)
        self.fc2 = nn.Linear(self.num_feats1, self.num_feats2)
        self.fc2_1 = nn.Linear(self.num_feats2, self.num_feats2)
        self.fc3 = nn.Linear(self.num_feats2, out_dims * softmax_bins)

        # Per-env dropout masks (approximates per-env diverse networks)
        self.register_buffer(
            "dropout_masks1",
            torch.ones(num_envs, self.num_feats1, device=device),
        )
        self.register_buffer(
            "dropout_masks2",
            torch.ones(num_envs, self.num_feats2, device=device),
        )

        # Bin centers for converting softmax to continuous action
        self.register_buffer(
            "bin_centers",
            torch.linspace(-1.0, 1.0, softmax_bins, device=device),
        )

        # Move to device and initialise
        self.to(device)
        self._refresh()

    def _refresh(self):
        """Re-initialise weights and dropout masks."""
        self._init_weights()
        self.eval()
        self.refresh_dropout_masks()

    def _init_weights(self):
        """Kaiming uniform initialisation for all layers."""
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc1_1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc2_1.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def refresh_dropout_masks(self):
        """Resample dropout masks with random probabilities."""
        dropout_probs = torch.rand(2)
        self.dropout_masks1.copy_(
            torch.bernoulli(
                torch.ones(self.num_envs, self.num_feats1), p=dropout_probs[0]
            )
        )
        self.dropout_masks2.copy_(
            torch.bernoulli(
                torch.ones(self.num_envs, self.num_feats2), p=dropout_probs[1]
            )
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: input → softmax distribution over bins per output dim.

        Args:
            x: Input tensor [num_envs, in_dims]

        Returns:
            Softmax probabilities [num_envs, out_dims, softmax_bins]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout_masks1 * self.fc1_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_masks2 * self.fc2_1(x)
        x = self.fc3(x)
        x = x.view(-1, self.out_dims, self.softmax_bins)
        return F.softmax(x, dim=-1)

    @torch.no_grad()
    def get_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """Get continuous action perturbation from RNA.

        Converts softmax distribution to continuous value via weighted
        sum with bin centers.

        Args:
            x: Input tensor [num_envs, in_dims] (dof_pos + object_pose)

        Returns:
            Action perturbation [num_envs, out_dims] in [-1, 1]
        """
        softmax_out = self.forward(x)
        # Weighted sum: [N, out, bins] * [bins] -> [N, out]
        return torch.sum(softmax_out * self.bin_centers, dim=-1)
