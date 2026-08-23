from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional
import random

import numpy as np
import torch
import torch.nn as nn

from interfaces.base_agent import BaseAgent


Transition = namedtuple(
    "Transition",
    [
        "state",
        "active_mask",
        "cluster_actions",
        "mig_split_actions",
        "mig_active_mask",
        "reward",
        "next_state",
        "next_active_mask",
        "next_available_migs",
        "done",
        "phase_id",
    ],
)


class ReplayBuffer:
    def __init__(self, capacity: int, phase_balanced: bool = True) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=int(capacity))
        self.phase_balanced = bool(phase_balanced)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.buffer):
            raise ValueError("batch_size exceeds replay buffer size")
        if not self.phase_balanced:
            return random.sample(list(self.buffer), batch_size)

        all_items = list(self.buffer)
        by_phase = {phase: [x for x in all_items if int(x.phase_id) == phase] for phase in range(3)}
        quota = batch_size // 3
        selected: list[Transition] = []
        selected_ids: set[int] = set()
        for phase in range(3):
            take = min(quota, len(by_phase[phase]))
            if take:
                chosen = random.sample(by_phase[phase], take)
                selected.extend(chosen)
                selected_ids.update(id(x) for x in chosen)
        remaining = batch_size - len(selected)
        if remaining:
            pool = [x for x in all_items if id(x) not in selected_ids]
            selected.extend(random.sample(pool, remaining))
        random.shuffle(selected)
        return selected


class PCSFLQNetwork(nn.Module):
    """Structured Q-network: client→MIG head + MIG→U-split head."""

    def __init__(
        self,
        state_dim: int,
        max_clients: int,
        max_migs: int,
        num_split_pairs: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.max_clients = int(max_clients)
        self.max_migs = int(max_migs)
        self.num_split_pairs = int(num_split_pairs)
        input_dim = self.max_clients * (int(state_dim) + 1)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cluster_head = nn.Linear(hidden_dim, self.max_clients * self.max_migs)
        self.mig_split_head = nn.Linear(hidden_dim, self.max_migs * self.num_split_pairs)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, padded_states: torch.Tensor, active_masks: torch.Tensor):
        active_feature = active_masks.unsqueeze(-1).to(dtype=padded_states.dtype)
        x = torch.cat([padded_states, active_feature], dim=-1).reshape(padded_states.shape[0], -1)
        hidden = self.backbone(x)
        cluster_q = self.cluster_head(hidden).view(-1, self.max_clients, self.max_migs)
        mig_split_q = self.mig_split_head(hidden).view(-1, self.max_migs, self.num_split_pairs)
        return cluster_q, mig_split_q


@dataclass
class _LastDecision:
    state: np.ndarray
    active_mask: np.ndarray
    cluster_actions: np.ndarray
    mig_split_actions: np.ndarray
    mig_active_mask: np.ndarray
    phase_id: int


class PCSFLAgent(BaseAgent):
    """Final PCSFL policy used by the project.

    The agent enforces cluster-level split consistency structurally, applies an exact
    client-capacity projection by default, and learns a joint system value from the
    client→MIG and MIG→split heads.
    """

    def __init__(
        self,
        state_dim: int = 15,
        max_clients: int = 20,
        max_migs: int = 7,
        num_layers: int = 7,
        hidden_dim: int = 256,
        gamma: float = 0.0,
        learning_rate: float = 1e-4,
        replay_capacity: int = 20_000,
        batch_size: int = 48,
        target_sync_interval: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        capacity_slack: int = 0,
        phase_balanced_replay: bool = True,
        use_joint_q: bool = True,
        device: Optional[str | torch.device] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(agent_name="PCSFL")
        self.state_dim = int(state_dim)
        self.max_clients = int(max_clients)
        self.max_migs = int(max_migs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.target_sync_interval = int(target_sync_interval)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.capacity_slack = int(capacity_slack)
        self.phase_balanced_replay = bool(phase_balanced_replay)
        self.use_joint_q = bool(use_joint_q)
        self.training_enabled = True
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.split_pairs = [(l1, l2) for l1 in range(self.num_layers) for l2 in range(l1, self.num_layers)]
        self.online_net = PCSFLQNetwork(
            self.state_dim, self.max_clients, self.max_migs, len(self.split_pairs), self.hidden_dim
        ).to(self.device)
        self.target_net = PCSFLQNetwork(
            self.state_dim, self.max_clients, self.max_migs, len(self.split_pairs), self.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity, phase_balanced=self.phase_balanced_replay)
        self.gradient_steps = 0
        self._last_decision: Optional[_LastDecision] = None

    def _prepare_state(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape (N, {self.state_dim})")
        n = state.shape[0]
        if n > self.max_clients:
            raise ValueError(f"N={n} exceeds max_clients={self.max_clients}")
        padded = np.zeros((self.max_clients, self.state_dim), dtype=np.float32)
        active = np.zeros(self.max_clients, dtype=np.float32)
        if n:
            padded[:n] = state
            active[:n] = 1.0
        return padded, active

    @staticmethod
    def _infer_phase_id(state: np.ndarray, active_mask: np.ndarray) -> int:
        n = int(active_mask.sum())
        if n == 0:
            return 0
        bandwidth_norm = float(state[0, -3])
        mig_norm = float(state[0, -2])
        if bandwidth_norm < 0.5:
            return 2
        if mig_norm >= 5.0 / 7.0 - 1e-6:
            return 1
        return 0

    def _capacity_vector(self, n_clients: int, available_migs: int, scores: np.ndarray) -> np.ndarray:
        m = int(available_migs)
        if self.capacity_slack > 0:
            return np.full(m, int(np.ceil(n_clients / m)) + self.capacity_slack, dtype=np.int64)
        base, remainder = divmod(n_clients, m)
        capacities = np.full(m, base, dtype=np.int64)
        if remainder:
            aggregate = scores[:n_clients, :m].mean(axis=0)
            extra = np.argsort(-aggregate)[:remainder]
            capacities[extra] += 1
        return capacities

    def _select_clusters(self, q: np.ndarray, n_clients: int, available_migs: int) -> np.ndarray:
        scores = q[:n_clients, :available_migs].copy()
        if self.training_enabled:
            for client in range(n_clients):
                if self.rng.random() < self.epsilon:
                    scores[client] = self.rng.normal(size=available_migs)
        capacities = self._capacity_vector(n_clients, available_migs, scores)
        actions = np.full(n_clients, -1, dtype=np.int64)
        remaining = capacities.copy()
        pairs = [
            (float(scores[i, j]), i, j)
            for i in range(n_clients)
            for j in range(available_migs)
        ]
        pairs.sort(reverse=True, key=lambda x: x[0])
        for _, client, mig in pairs:
            if actions[client] >= 0 or remaining[mig] <= 0:
                continue
            actions[client] = mig
            remaining[mig] -= 1
        # Slack>0 can leave unassigned clients if greedy capacity was exhausted unexpectedly.
        for client in np.where(actions < 0)[0]:
            available = np.where(remaining > 0)[0]
            mig = int(available[np.argmax(scores[client, available])]) if available.size else int(np.argmax(scores[client]))
            actions[client] = mig
            if remaining[mig] > 0:
                remaining[mig] -= 1
        return actions

    @torch.no_grad()
    def step(self, active_clients_state: np.ndarray, available_migs: int):
        state = np.asarray(active_clients_state, dtype=np.float32)
        n_clients = state.shape[0]
        if not 1 <= int(available_migs) <= self.max_migs:
            raise ValueError("available_migs out of range")
        padded, active_mask = self._prepare_state(state)
        state_t = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(active_mask).unsqueeze(0).to(self.device)
        cluster_q, mig_split_q = self.online_net(state_t, mask_t)
        cluster_q_np = cluster_q.squeeze(0).cpu().numpy()
        split_q_np = mig_split_q.squeeze(0).cpu().numpy()

        cluster_actions_n = self._select_clusters(cluster_q_np, n_clients, int(available_migs))
        cluster_actions = np.zeros(self.max_clients, dtype=np.int64)
        cluster_actions[:n_clients] = cluster_actions_n
        mig_active_mask = np.zeros(self.max_migs, dtype=np.float32)
        mig_split_actions = np.zeros(self.max_migs, dtype=np.int64)
        l1_choices = np.zeros(n_clients, dtype=np.int64)
        l2_choices = np.zeros(n_clients, dtype=np.int64)

        for mig in range(int(available_migs)):
            idx = np.where(cluster_actions_n == mig)[0]
            if idx.size == 0:
                continue
            mig_active_mask[mig] = 1.0
            if self.training_enabled and self.rng.random() < self.epsilon:
                split_idx = int(self.rng.integers(0, len(self.split_pairs)))
            else:
                split_idx = int(np.argmax(split_q_np[mig]))
            mig_split_actions[mig] = split_idx
            pair = self.split_pairs[split_idx]
            l1_choices[idx] = pair[0]
            l2_choices[idx] = pair[1]

        phase_id = self._infer_phase_id(padded, active_mask)
        self._last_decision = _LastDecision(
            state=padded.copy(),
            active_mask=active_mask.copy(),
            cluster_actions=cluster_actions.copy(),
            mig_split_actions=mig_split_actions.copy(),
            mig_active_mask=mig_active_mask.copy(),
            phase_id=phase_id,
        )
        bw_weights = np.ones(n_clients, dtype=np.float32)
        return cluster_actions_n, l1_choices, l2_choices, bw_weights

    def observe(self, reward: float, next_state: np.ndarray, next_available_migs: int, done: bool = True) -> Optional[float]:
        if self._last_decision is None:
            raise RuntimeError("step() must be called before observe()")
        next_padded, next_active = self._prepare_state(next_state)
        transition = Transition(
            state=self._last_decision.state,
            active_mask=self._last_decision.active_mask,
            cluster_actions=self._last_decision.cluster_actions,
            mig_split_actions=self._last_decision.mig_split_actions,
            mig_active_mask=self._last_decision.mig_active_mask,
            reward=float(reward),
            next_state=next_padded,
            next_active_mask=next_active,
            next_available_migs=int(next_available_migs),
            done=float(done),
            phase_id=int(self._last_decision.phase_id),
        )
        self.replay_buffer.push(transition)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.train_step()

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        active_masks = torch.as_tensor(np.stack([x.active_mask for x in batch]), dtype=torch.float32, device=self.device)
        cluster_actions = torch.as_tensor(np.stack([x.cluster_actions for x in batch]), dtype=torch.long, device=self.device)
        split_actions = torch.as_tensor(np.stack([x.mig_split_actions for x in batch]), dtype=torch.long, device=self.device)
        mig_masks = torch.as_tensor(np.stack([x.mig_active_mask for x in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([x.done for x in batch], dtype=torch.float32, device=self.device)

        cluster_q, split_q = self.online_net(states, active_masks)
        chosen_cluster_q = cluster_q.gather(-1, cluster_actions.unsqueeze(-1)).squeeze(-1)
        chosen_split_q = split_q.gather(-1, split_actions.unsqueeze(-1)).squeeze(-1)

        if self.gamma == 0.0:
            target = rewards
        else:
            next_states = torch.as_tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=self.device)
            next_masks = torch.as_tensor(np.stack([x.next_active_mask for x in batch]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_cluster_online, next_split_online = self.online_net(next_states, next_masks)
                next_cluster_target, next_split_target = self.target_net(next_states, next_masks)
                next_cluster_a = next_cluster_online.argmax(dim=-1)
                next_split_a = next_split_online.argmax(dim=-1)
                next_cluster_v = next_cluster_target.gather(-1, next_cluster_a.unsqueeze(-1)).squeeze(-1)
                next_split_v = next_split_target.gather(-1, next_split_a.unsqueeze(-1)).squeeze(-1)
                cluster_v = (next_cluster_v * next_masks).sum(1) / next_masks.sum(1).clamp_min(1.0)
                next_mig_mask = torch.zeros_like(next_split_v)
                for row, item in enumerate(batch):
                    next_mig_mask[row, : int(item.next_available_migs)] = 1.0
                split_v = (next_split_v * next_mig_mask).sum(1) / next_mig_mask.sum(1).clamp_min(1.0)
                bootstrap = 0.5 * (cluster_v + split_v)
                target = rewards + self.gamma * (1.0 - dones) * bootstrap

        if self.use_joint_q:
            cluster_total = (chosen_cluster_q * active_masks).sum(1) / active_masks.sum(1).clamp_min(1.0)
            split_total = (chosen_split_q * mig_masks).sum(1) / mig_masks.sum(1).clamp_min(1.0)
            q_total = 0.5 * (cluster_total + split_total)
            loss = nn.functional.smooth_l1_loss(q_total, target)
        else:
            cluster_target = target.unsqueeze(1).expand_as(chosen_cluster_q)
            split_target = target.unsqueeze(1).expand_as(chosen_split_q)
            cden = active_masks.sum().clamp_min(1.0)
            sden = mig_masks.sum().clamp_min(1.0)
            cluster_loss = (nn.functional.smooth_l1_loss(chosen_cluster_q, cluster_target, reduction="none") * active_masks).sum() / cden
            split_loss = (nn.functional.smooth_l1_loss(chosen_split_q, split_target, reduction="none") * mig_masks).sum() / sden
            loss = cluster_loss + split_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def set_training(self, enabled: bool) -> None:
        self.training_enabled = bool(enabled)
        self.online_net.train(enabled)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "gradient_steps": self.gradient_steps,
                "config": {
                    "max_clients": self.max_clients,
                    "max_migs": self.max_migs,
                    "num_layers": self.num_layers,
                    "hidden_dim": self.hidden_dim,
                    "gamma": self.gamma,
                    "capacity_slack": self.capacity_slack,
                    "phase_balanced_replay": self.phase_balanced_replay,
                    "use_joint_q": self.use_joint_q,
                },
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = False) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(payload["online_net"])
        self.target_net.load_state_dict(payload.get("target_net", payload["online_net"]))
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon = float(payload.get("epsilon", self.epsilon_end))
        self.gradient_steps = int(payload.get("gradient_steps", 0))
