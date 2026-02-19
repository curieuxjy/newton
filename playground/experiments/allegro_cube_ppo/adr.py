"""Vectorised Automatic Domain Randomisation (ADR) for Newton environments.

Reference: DextrEme paper Section 4
Original: IsaacGymEnvs/isaacgymenvs/tasks/dextreme/adr_vec_task.py

ADR automatically adjusts domain randomisation ranges:
- 60% of envs are "rollout workers" sampling from current ADR ranges
- 40% are "boundary workers" evaluating specific boundary values
- Performance queues (N=256) track consecutive successes per boundary
- Boundaries widen when performance > threshold_high (20)
- Boundaries tighten when performance < threshold_low (5)
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


class WorkerMode:
    """Environment worker types for ADR."""

    ROLLOUT = 0  # Sample from current ADR range
    BOUNDARY = 1  # Evaluate at specific boundary


@dataclass
class ADRParamDef:
    """Definition of a single ADR parameter.

    Attributes:
        name: Parameter identifier
        init_range: [lower, upper] initial bounds (start of ADR)
        limits: [min_allowed, max_allowed] hard limits for boundary expansion
        delta: Step size for boundary changes
        delta_style: "additive" or "multiplicative"
    """

    name: str
    init_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    limits: list[float] = field(default_factory=lambda: [0.0, 1.0])
    delta: float = 0.1
    delta_style: str = "additive"

    def __post_init__(self):
        # Current range (mutable, adjusted by ADR)
        self.range = list(self.init_range)


def make_default_adr_params(
    hand_stiffness: float = 40.0,
    hand_damping: float = 2.0,
    cube_mass: float = 0.1,
    hand_friction: float = 1.2,
    cube_friction: float = 1.2,
) -> list[ADRParamDef]:
    """Create default ADR parameter definitions for Allegro hand cube rotation.

    Initial ranges start at nominal values (no randomization).
    ADR will gradually widen ranges as the policy improves.

    Args:
        hand_stiffness: Nominal joint PD stiffness
        hand_damping: Nominal joint PD damping
        cube_mass: Nominal cube mass [kg]
        hand_friction: Nominal hand friction coefficient
        cube_friction: Nominal cube friction coefficient
    """
    return [
        ADRParamDef(
            name="hand_stiffness",
            init_range=[hand_stiffness, hand_stiffness],
            limits=[5.0, 200.0],
            delta=5.0,
        ),
        ADRParamDef(
            name="hand_damping",
            init_range=[hand_damping, hand_damping],
            limits=[0.1, 20.0],
            delta=0.5,
        ),
        ADRParamDef(
            name="cube_mass",
            init_range=[cube_mass, cube_mass],
            limits=[0.01, 0.5],
            delta=0.02,
        ),
        ADRParamDef(
            name="hand_friction",
            init_range=[hand_friction, hand_friction],
            limits=[0.1, 3.0],
            delta=0.1,
        ),
        ADRParamDef(
            name="cube_friction",
            init_range=[cube_friction, cube_friction],
            limits=[0.1, 3.0],
            delta=0.1,
        ),
        ADRParamDef(
            name="obs_noise",
            init_range=[0.0, 0.0],
            limits=[0.0, 0.5],
            delta=0.02,
        ),
        ADRParamDef(
            name="action_noise",
            init_range=[0.0, 0.0],
            limits=[0.0, 0.3],
            delta=0.01,
        ),
    ]


class ADRManager:
    """Vectorised Automatic Domain Randomisation manager.

    Implements the ADR algorithm from the DextrEme paper:
    1. Environments are split into rollout workers (60%) and boundary workers (40%)
    2. Boundary workers test specific parameter boundaries
    3. Performance queues accumulate consecutive success counts
    4. When queues fill (N=256), boundaries are widened or tightened

    Args:
        num_envs: Number of parallel environments
        device: Torch device string
        params: List of ADR parameter definitions
        boundary_fraction: Fraction of envs dedicated to boundary evaluation (40%)
        queue_length: Performance queue capacity (N=256)
        threshold_high: Widen boundary when mean performance > this (20)
        threshold_low: Tighten boundary when mean performance < this (5)
        clear_other_queues: Clear all queues when any boundary changes
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        params: list[ADRParamDef] | None = None,
        boundary_fraction: float = 0.4,
        queue_length: int = 256,
        threshold_high: float = 20.0,
        threshold_low: float = 5.0,
        clear_other_queues: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.boundary_fraction = boundary_fraction
        self.queue_length = queue_length
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.clear_other_queues = clear_other_queues

        # Deep copy params so each ADRManager has independent state
        self.params = params if params is not None else make_default_adr_params()
        self.num_params = len(self.params)
        self.param_names = [p.name for p in self.params]

        # Worker assignments: which type is each env?
        self.worker_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        # ADR modes: which boundary does each boundary worker evaluate?
        # mode = 2*n → lower bound of param n, mode = 2*n+1 → upper bound
        self.adr_modes = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Performance queues: 2 per param (lower + upper boundary)
        self.queues: list[deque] = [deque(maxlen=queue_length) for _ in range(2 * self.num_params)]

        # Current sampled values per env per param
        self.values: dict[str, torch.Tensor] = {}
        for p in self.params:
            self.values[p.name] = torch.zeros(num_envs, device=device)

        # All start as rollout workers
        self.worker_types.fill_(WorkerMode.ROLLOUT)

        # Metrics
        self.total_boundary_changes = 0
        self.rollout_perf_ema = 0.0
        self.rollout_perf_alpha = 0.3

    def recycle_envs(self, env_ids: torch.Tensor):
        """Reassign finished environments to new worker types.

        Paper: 40% become boundary workers, 60% become rollout workers.
        Each boundary worker is randomly assigned to evaluate one specific
        boundary (lower or upper) of one specific parameter.
        """
        if len(env_ids) == 0:
            return

        rand = torch.rand(len(env_ids), device=self.device)
        rollout_threshold = 1.0 - self.boundary_fraction  # 0.6

        new_types = torch.where(
            rand < rollout_threshold,
            torch.full_like(env_ids, WorkerMode.ROLLOUT),
            torch.full_like(env_ids, WorkerMode.BOUNDARY),
        )
        self.worker_types[env_ids] = new_types

        # Randomly assign which boundary each boundary worker evaluates
        self.adr_modes[env_ids] = torch.randint(0, self.num_params * 2, (len(env_ids),), device=self.device)

    def update(self, env_ids: torch.Tensor, objective: torch.Tensor):
        """Main ADR update: add objectives to queues, update boundaries.

        Called when environments finish episodes (are about to be reset).
        The objective is the consecutive successes count for each env.

        Args:
            env_ids: IDs of environments finishing episodes
            objective: Per-env objective values (shape [num_envs])
        """
        if len(env_ids) == 0:
            return

        any_changed = False

        # Track rollout worker performance (EMA)
        rollout_mask = self.worker_types[env_ids] == WorkerMode.ROLLOUT
        if rollout_mask.any():
            rollout_obj = objective[env_ids[rollout_mask]].float().mean().item()
            self.rollout_perf_ema = self.rollout_perf_alpha * rollout_obj + (1.0 - self.rollout_perf_alpha) * self.rollout_perf_ema

        for n, param in enumerate(self.params):
            low_idx = 2 * n
            high_idx = 2 * n + 1

            # Find boundary workers for this parameter among the finishing envs
            boundary_mask = self.worker_types[env_ids] == WorkerMode.BOUNDARY
            low_mask = boundary_mask & (self.adr_modes[env_ids] == low_idx)
            high_mask = boundary_mask & (self.adr_modes[env_ids] == high_idx)

            # Add objectives to queues
            if low_mask.any():
                obj_vals = objective[env_ids[low_mask]].float().cpu().numpy().tolist()
                self.queues[low_idx].extend(obj_vals)
            if high_mask.any():
                obj_vals = objective[env_ids[high_mask]].float().cpu().numpy().tolist()
                self.queues[high_idx].extend(obj_vals)

            # --- Lower boundary decision ---
            changed_low = False
            if len(self.queues[low_idx]) >= self.queue_length:
                mean_low = float(np.mean(self.queues[low_idx]))
                if mean_low < self.threshold_low:
                    # Performance too low → tighten: move lower bound toward init
                    new_val, did_change = self._modify_param(param.range[0], "up", param, limit=param.init_range[0])
                    if did_change:
                        param.range[0] = new_val
                        changed_low = True
                elif mean_low > self.threshold_high:
                    # Performance high → widen: move lower bound outward
                    new_val, did_change = self._modify_param(param.range[0], "down", param, limit=param.limits[0])
                    if did_change:
                        param.range[0] = new_val
                        changed_low = True
                if changed_low:
                    self.queues[low_idx].clear()
                    # Convert these boundary workers back to rollout
                    low_workers = (self.worker_types == WorkerMode.BOUNDARY) & (self.adr_modes == low_idx)
                    self.worker_types[low_workers] = WorkerMode.ROLLOUT

            # --- Upper boundary decision ---
            changed_high = False
            if len(self.queues[high_idx]) >= self.queue_length:
                mean_high = float(np.mean(self.queues[high_idx]))
                if mean_high < self.threshold_low:
                    # Performance too low → tighten: move upper bound toward init
                    new_val, did_change = self._modify_param(param.range[1], "down", param, limit=param.init_range[1])
                    if did_change:
                        param.range[1] = new_val
                        changed_high = True
                elif mean_high > self.threshold_high:
                    # Performance high → widen: move upper bound outward
                    new_val, did_change = self._modify_param(param.range[1], "up", param, limit=param.limits[1])
                    if did_change:
                        param.range[1] = new_val
                        changed_high = True
                if changed_high:
                    self.queues[high_idx].clear()
                    high_workers = (self.worker_types == WorkerMode.BOUNDARY) & (self.adr_modes == high_idx)
                    self.worker_types[high_workers] = WorkerMode.ROLLOUT

            if changed_low or changed_high:
                any_changed = True
                self.total_boundary_changes += 1

        # Optionally clear all queues when any boundary changes
        if any_changed and self.clear_other_queues:
            for q in self.queues:
                q.clear()
            self.worker_types[self.worker_types == WorkerMode.BOUNDARY] = WorkerMode.ROLLOUT

    def sample(self, param_name: str, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample ADR parameter values for specified environments.

        Rollout workers: uniform sample from current [lower, upper] range.
        Boundary workers testing THIS param: use exact boundary value.
        Boundary workers testing OTHER params: uniform sample from current range.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = self.param_names.index(param_name)
        param = self.params[n]
        low_idx = 2 * n
        high_idx = 2 * n + 1

        result = torch.zeros(len(env_ids), device=self.device)

        types = self.worker_types[env_ids]
        modes = self.adr_modes[env_ids]

        # Boundary workers testing THIS param's boundaries
        boundary_low = (types == WorkerMode.BOUNDARY) & (modes == low_idx)
        boundary_high = (types == WorkerMode.BOUNDARY) & (modes == high_idx)
        # Everyone else: uniform sample from current range
        uniform_mask = ~boundary_low & ~boundary_high

        lo, hi = param.range
        num_uniform = uniform_mask.sum().item()
        if num_uniform > 0:
            if abs(hi - lo) < 1e-9:
                result[uniform_mask] = lo
            else:
                result[uniform_mask] = torch.rand(num_uniform, device=self.device) * (hi - lo) + lo

        if boundary_low.any():
            result[boundary_low] = lo
        if boundary_high.any():
            result[boundary_high] = hi

        self.values[param_name][env_ids] = result
        return result

    def sample_all(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Sample all ADR parameters for specified environments."""
        return {p.name: self.sample(p.name, env_ids) for p in self.params}

    def _modify_param(self, value: float, direction: str, param: ADRParamDef, limit: float | None) -> tuple[float, bool]:
        """Modify a parameter boundary by delta in the given direction."""
        if param.delta_style == "additive":
            new_val = value + param.delta if direction == "up" else value - param.delta
        elif param.delta_style == "multiplicative":
            new_val = value * param.delta if direction == "up" else value / param.delta
        else:
            return value, False

        # Clamp to limit
        if limit is not None:
            new_val = min(new_val, limit) if direction == "up" else max(new_val, limit)

        return new_val, abs(new_val - value) > 1e-9

    def get_metrics(self) -> dict[str, float]:
        """Return ADR metrics for logging."""
        metrics: dict[str, float] = {
            "adr/rollout_perf": self.rollout_perf_ema,
            "adr/boundary_changes": float(self.total_boundary_changes),
        }

        total_nats = 0.0
        for n, param in enumerate(self.params):
            lo, hi = param.range
            width = max(hi - lo, 1e-3)
            total_nats += np.log(width)

            metrics[f"adr/{param.name}/lower"] = lo
            metrics[f"adr/{param.name}/upper"] = hi
            metrics[f"adr/{param.name}/width"] = hi - lo

            low_q = self.queues[2 * n]
            high_q = self.queues[2 * n + 1]
            if len(low_q) > 0:
                metrics[f"adr/{param.name}/low_q_mean"] = float(np.mean(low_q))
            if len(high_q) > 0:
                metrics[f"adr/{param.name}/high_q_mean"] = float(np.mean(high_q))

        metrics["adr/npd"] = total_nats / max(self.num_params, 1)  # Nats per dimension

        return metrics

    def state_dict(self) -> dict:
        """Get ADR state for checkpointing."""
        return {
            "param_ranges": {p.name: p.range[:] for p in self.params},
            "queues": [list(q) for q in self.queues],
            "worker_types": self.worker_types.cpu(),
            "adr_modes": self.adr_modes.cpu(),
            "total_boundary_changes": self.total_boundary_changes,
            "rollout_perf_ema": self.rollout_perf_ema,
        }

    def load_state_dict(self, state: dict):
        """Restore ADR state from checkpoint."""
        for p in self.params:
            if p.name in state["param_ranges"]:
                p.range = list(state["param_ranges"][p.name])

        for i, q_data in enumerate(state["queues"]):
            self.queues[i].clear()
            self.queues[i].extend(q_data)

        self.worker_types.copy_(state["worker_types"].to(self.device))
        self.adr_modes.copy_(state["adr_modes"].to(self.device))
        self.total_boundary_changes = state["total_boundary_changes"]
        self.rollout_perf_ema = state["rollout_perf_ema"]
