"""Student distillation for DEXTRAH-G depth policy.

DEXTRAH-G Stage 2: Distill the privileged teacher into a student policy
that uses depth images + proprioception (no object state).

The student learns to imitate teacher actions via:
    L = L_action + beta * L_pos
where:
    L_action = weighted L2 between student and teacher actions (w = 1/sigma_teacher^2)
    L_pos = L2 prediction of object position from depth (auxiliary task)
    beta: 1.0 -> 0.0 over 15k iterations

Run with:
    uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.distill \
        --teacher-checkpoint checkpoints/teacher_*/best.pt

Reference: DEXTRAH-G distillation.py
"""

import argparse
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .config import DistillConfig, EnvConfig, StudentConfig, TeacherPPOConfig, TrainConfig
from .env import FrankaAllegroGraspEnv
from .networks import StudentNetwork, TeacherActorCritic


class DistillationTrainer:
    """Distill teacher policy into student depth policy."""

    def __init__(
        self,
        env: FrankaAllegroGraspEnv,
        teacher: TeacherActorCritic,
        config: TrainConfig,
    ):
        self.env = env
        self.teacher = teacher
        self.teacher.eval()
        self.config = config
        self.student_cfg = config.student
        self.distill_cfg = config.distill
        self.device = config.device

        # Create student network
        self.student = StudentNetwork(
            num_proprio_obs=env.num_student_obs,
            num_actions=env.num_actions,
            depth_channels=self.student_cfg.cnn_channels,
            depth_kernel_sizes=self.student_cfg.cnn_kernel_sizes,
            depth_strides=self.student_cfg.cnn_strides,
            cnn_output_dim=self.student_cfg.cnn_output_dim,
            cnn_use_layer_norm=self.student_cfg.cnn_use_layer_norm,
            lstm_units=self.student_cfg.lstm_units,
            lstm_layers=self.student_cfg.lstm_layers,
            lstm_layer_norm=self.student_cfg.lstm_layer_norm,
            mlp_dims=self.student_cfg.mlp_dims,
            aux_mlp_dims=self.student_cfg.aux_mlp_dims,
            aux_output_dim=self.student_cfg.aux_output_dim,
            activation=self.student_cfg.activation,
            init_sigma=self.student_cfg.init_sigma,
        ).to(self.device)

        self.optimizer = optim.Adam(self.student.parameters(), lr=self.distill_cfg.learning_rate)

        # LR scheduler with warmup
        def lr_lambda(step):
            if step < self.distill_cfg.warmup_steps:
                return step / max(self.distill_cfg.warmup_steps, 1)
            return 1.0

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # LSTM hidden states
        self.teacher_actor_hidden = None
        self.student_hidden = None

    def get_beta(self, iteration: int) -> float:
        """Compute auxiliary loss weight schedule."""
        if iteration >= self.distill_cfg.beta_decay_iteration:
            return self.distill_cfg.beta_final
        t = iteration / self.distill_cfg.beta_decay_iteration
        return self.distill_cfg.beta_initial * (1.0 - t) + self.distill_cfg.beta_final * t

    def train_step(self, iteration: int) -> dict[str, float]:
        """Run one distillation step: collect teacher action, train student."""
        # Get current observations
        teacher_obs = self.env.teacher_obs_buf
        student_obs = self.env.student_obs_buf
        depth = self.env.depth_obs_buf

        if depth is None:
            raise ValueError("Depth sensor must be enabled for distillation")

        # Get teacher action (privileged, no grad)
        with torch.no_grad():
            teacher_result = self.teacher.get_action_and_value(
                actor_obs=teacher_obs,
                critic_obs=teacher_obs,
                actor_hidden=self.teacher_actor_hidden,
                critic_hidden=None,
            )
            teacher_action = teacher_result["action"]
            teacher_sigma = teacher_result["action_std"]
            self.teacher_actor_hidden = teacher_result["actor_hidden"]

        # Get student prediction
        student_result = self.student.get_action_and_value(
            depth_image=depth,
            proprio_obs=student_obs,
            hidden=self.student_hidden,
        )
        student_action_mean = student_result["action_mean"]
        aux_pred = student_result["aux_pred"]
        self.student_hidden = tuple(h.detach() for h in student_result["hidden"])

        # === Distillation Loss ===

        # 1. Action loss: weighted L2 (w = 1/sigma_teacher^2)
        if self.distill_cfg.use_inverse_variance_weighting:
            weight = 1.0 / (teacher_sigma**2 + 1e-6)
            action_loss = (weight * (student_action_mean - teacher_action)**2).mean()
        else:
            action_loss = ((student_action_mean - teacher_action)**2).mean()

        # 2. Auxiliary loss: object position prediction
        # Ground truth: cube position from teacher obs (privileged)
        # In teacher_obs, object position starts at index num_student_obs
        cube_pos_gt = teacher_obs[:, self.env.num_student_obs:self.env.num_student_obs + 3]
        aux_loss = ((aux_pred - cube_pos_gt)**2).mean()

        # 3. Total loss with beta schedule
        beta = self.get_beta(iteration)
        total_loss = action_loss + beta * aux_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Step environment with teacher action (to generate new observations)
        with torch.no_grad():
            action_to_apply = torch.clamp(teacher_action, -1.0, 1.0)
        _, _, done, _ = self.env.step(action_to_apply)

        # Reset hidden states for done environments
        done_indices = torch.where(done)[0]
        if len(done_indices) > 0:
            if self.teacher_actor_hidden is not None:
                self.teacher_actor_hidden[0][:, done_indices] = 0
                self.teacher_actor_hidden[1][:, done_indices] = 0
            if self.student_hidden is not None:
                self.student_hidden[0][:, done_indices] = 0
                self.student_hidden[1][:, done_indices] = 0

        return {
            "total_loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "aux_loss": aux_loss.item(),
            "beta": beta,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train(self):
        """Main distillation training loop."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"student_{self.config.experiment_name}_{timestamp}"
        log_dir = Path("runs") / run_name
        checkpoint_dir = Path(self.config.checkpoint_dir) / run_name

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir)

        # W&B
        if self.config.wandb.enabled:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    name=run_name,
                    config=asdict(self.config),
                    tags=self.config.wandb.tags + ["student", "distillation"],
                    mode=self.config.wandb.mode,
                )
            except ImportError:
                self.config.wandb.enabled = False

        # Initialize
        self.env.reset()

        # Initialize hidden states
        self.teacher_actor_hidden = self.teacher.init_hidden(self.env.num_envs, self.device)["actor"]
        self.student_hidden = self.student.init_hidden(self.env.num_envs, self.device)

        print("=" * 60)
        print("DEXTRAH-G Student Distillation (Depth Policy)")
        print("=" * 60)
        print(f"  Envs: {self.env.num_envs}")
        print(f"  Student proprio obs: {self.env.num_student_obs}D")
        print(f"  Depth: {self.env.config.depth_height}x{self.env.config.depth_width}")
        print(f"  CNN output: {self.student_cfg.cnn_output_dim}D")
        print(f"  LSTM: {self.student_cfg.lstm_units}")
        print(f"  Max iterations: {self.distill_cfg.max_iterations}")
        print(f"  Checkpoints: {checkpoint_dir}")
        print("=" * 60)

        start_time = time.time()
        best_loss = float("inf")
        running_loss = 0.0

        for iteration in range(1, self.distill_cfg.max_iterations + 1):
            metrics = self.train_step(iteration)
            running_loss += metrics["total_loss"]

            if iteration % self.distill_cfg.log_interval == 0:
                avg_loss = running_loss / self.distill_cfg.log_interval
                elapsed = time.time() - start_time
                fps = iteration * self.env.num_envs / elapsed

                print(
                    f"[iter {iteration:>7}] "
                    f"loss: {avg_loss:.4f} | "
                    f"action: {metrics['action_loss']:.4f} | "
                    f"aux: {metrics['aux_loss']:.4f} | "
                    f"beta: {metrics['beta']:.3f} | "
                    f"lr: {metrics['lr']:.2e} | "
                    f"FPS: {fps:.0f}"
                )

                writer.add_scalar("distill/total_loss", avg_loss, iteration)
                writer.add_scalar("distill/action_loss", metrics["action_loss"], iteration)
                writer.add_scalar("distill/aux_loss", metrics["aux_loss"], iteration)
                writer.add_scalar("distill/beta", metrics["beta"], iteration)
                writer.add_scalar("distill/lr", metrics["lr"], iteration)

                if self.config.wandb.enabled:
                    import wandb
                    wandb.log({f"distill/{k}": v for k, v in metrics.items()}, step=iteration)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save(checkpoint_dir / "best.pt")

                running_loss = 0.0

            if iteration % self.distill_cfg.save_interval == 0:
                self.save(checkpoint_dir / f"checkpoint_{iteration}.pt")

        self.save(checkpoint_dir / "final.pt")
        print(f"[INFO] Distillation complete! Final model: {checkpoint_dir / 'final.pt'}")

        writer.close()
        if self.config.wandb.enabled:
            import wandb
            wandb.finish()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "student": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])


def main():
    parser = argparse.ArgumentParser(description="DEXTRAH-G Student Distillation")

    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--student-checkpoint", type=str, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_config = EnvConfig(
        num_envs=args.num_envs,
        use_depth_sensor=True,  # Student uses depth
        use_fabric_actions=True,
    )

    distill_config = DistillConfig(
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    train_config = TrainConfig(
        seed=args.seed,
        env=env_config,
        distill=distill_config,
    )
    train_config.wandb.enabled = not args.no_wandb

    # Create environment (with depth)
    print("[INFO] Creating environment with depth sensor...")
    env = FrankaAllegroGraspEnv(env_config, device="cuda", headless=True)

    # Load teacher
    print(f"[INFO] Loading teacher from {args.teacher_checkpoint}...")
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location="cuda")

    # Reconstruct teacher config
    teacher_ppo = train_config.teacher
    teacher = TeacherActorCritic(
        num_actor_obs=env.num_teacher_obs,
        num_critic_obs=env.num_teacher_obs,
        num_actions=env.num_actions,
        actor_lstm_units=teacher_ppo.actor_lstm_units,
        critic_lstm_units=teacher_ppo.critic_lstm_units,
        actor_mlp_dims=teacher_ppo.actor_mlp_dims,
        critic_mlp_dims=teacher_ppo.critic_mlp_dims,
        lstm_layers=teacher_ppo.lstm_layers,
        lstm_layer_norm=teacher_ppo.lstm_layer_norm,
        activation=teacher_ppo.activation,
    ).to("cuda")
    teacher.load_state_dict(teacher_ckpt["actor_critic"])
    teacher.eval()

    # Create trainer
    print("[INFO] Creating distillation trainer...")
    trainer = DistillationTrainer(env=env, teacher=teacher, config=train_config)

    if args.student_checkpoint:
        print(f"[INFO] Loading student checkpoint: {args.student_checkpoint}")
        trainer.load(args.student_checkpoint)

    print("[INFO] Starting distillation...")
    trainer.train()


if __name__ == "__main__":
    main()
