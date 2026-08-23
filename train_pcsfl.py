from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from baselines.pcsfl import PCSFLAgent, TensorProfileCache, build_agent_state, evaluate_actions
from data.cifar_10_provider import CIFAR10NonIIDProvider
from envs.liquid_airan_env import LiquidAIRANEnv
from models.usfl_networks import ResNet18_USFL


# ============================================================
# 以下辅助函数只负责终端输出，不参与 PCSFL 的状态、动作、奖励和训练逻辑
# ============================================================

def _阶段名称(epoch: int) -> str:
    if epoch <= 50:
        return "阶段一：常态资源"
    if epoch <= 100:
        return "阶段二：车辆增加与 MIG 资源扩展"
    return "阶段三：低带宽通信拥塞"


def _格式化_loss(loss) -> str:
    return "经验池积累中" if loss is None else f"{loss:.6f}"


def _打印每轮信息(
    *,
    episode: int,
    episodes: int,
    epoch: int,
    rounds: int,
    row: dict,
) -> None:
    """每一轮都输出当前环境和核心性能指标。"""
    print(
        f"[训练周期 {episode:02d}/{episodes:02d} | "
        f"轮次 {epoch:03d}/{rounds:03d} | {_阶段名称(epoch)}] "
        f"客户端={row['n_clients']:2d}，"
        f"MIG={row['n_migs']}，"
        f"带宽={row['bandwidth_mhz']:.1f} MHz，"
        f"完整U型时延={row['full_total_delay_ms']:.3f} ms，"
        f"奖励={row['reward']:.5f}，"
        f"冲突率={row['conflict_rate'] * 100:.1f}%"
    )


def _打印十轮详细信息(
    *,
    episode: int,
    epoch: int,
    row: dict,
) -> None:
    """每 10 轮打印一次更详细的训练状态。"""
    print()
    print("=" * 88)
    print(
        f"【第 {episode:02d} 个训练周期 · 第 {epoch:03d} 轮详细状态】"
        f"  {_阶段名称(epoch)}"
    )
    print("-" * 88)
    print(
        f"环境状态：客户端数量={row['n_clients']}，"
        f"可用 MIG 数={row['n_migs']}，"
        f"系统带宽={row['bandwidth_mhz']:.1f} MHz，"
        f"环境种子={row['seed']}"
    )
    print(
        f"时延指标：完整U型总时延={row['full_total_delay_ms']:.3f} ms，"
        f"旧口径总时延={row['total_delay_ms']:.3f} ms"
    )
    print(
        f"时延分解：上行={row['uplink_delay_ms']:.3f} ms，"
        f"下行={row['downlink_delay_ms']:.3f} ms，"
        f"计算={row['comp_delay_ms']:.3f} ms，"
        f"通信参考值={row['tx_delay_ms']:.3f} ms"
    )
    print(
        f"资源分配：MIG负载={row['cluster_loads']}，"
        f"负载不均衡={row['load_imbalance']:.4f}"
    )
    print(f"切分决策：各MIG切分点={row['mig_splits']}")
    print(
        f"切分统计：平均 l1={row['mean_l1']:.3f}，"
        f"平均 l2={row['mean_l2']:.3f}"
    )
    print(
        f"张量统计：平均上传={row['mean_upload_bytes']:.1f} Bytes，"
        f"平均下行={row['mean_downlink_bytes']:.1f} Bytes"
    )
    print(
        f"训练状态：reward={row['reward']:.6f}，"
        f"loss={_格式化_loss(row['loss'])}，"
        f"epsilon={row['epsilon']:.4f}"
    )
    print(
        f"经验学习：ReplayBuffer={row['buffer_size']}，"
        f"累计梯度更新={row['gradient_steps']}，"
        f"全局训练步={row['global_step']}"
    )
    print("=" * 88)
    print()


def _打印五十轮阶段总结(
    *,
    episode: int,
    epoch: int,
    history: list[dict],
) -> None:
    """
    每 50 轮对当前训练周期最近 50 轮做统计。
    这里只读取 history 并打印，不修改任何训练数据。
    """
    start_epoch = epoch - 49
    block = [
        row
        for row in history
        if row["episode"] == episode and start_epoch <= row["epoch"] <= epoch
    ]

    if not block:
        return

    def 平均(field: str) -> float:
        return float(np.mean([float(row[field]) for row in block]))

    phase_index = (epoch - 1) // 50 + 1
    phase_title = _阶段名称(epoch)

    first_row = block[0]
    last_row = block[-1]

    print()
    print("#" * 88)
    print(
        f"【阶段总结：第 {episode:02d} 个训练周期 · "
        f"阶段 {phase_index} · 轮次 {start_epoch:03d}-{epoch:03d}】"
    )
    print(f"阶段定义：{phase_title}")
    print("-" * 88)
    print(
        f"环境概况：平均客户端数量={平均('n_clients'):.2f}，"
        f"平均可用 MIG={平均('n_migs'):.2f}，"
        f"平均带宽={平均('bandwidth_mhz'):.2f} MHz"
    )
    print(
        f"时延总结：平均完整U型时延={平均('full_total_delay_ms'):.3f} ms，"
        f"平均旧口径时延={平均('total_delay_ms'):.3f} ms"
    )
    print(
        f"时延分解：平均上行={平均('uplink_delay_ms'):.3f} ms，"
        f"平均下行={平均('downlink_delay_ms'):.3f} ms，"
        f"平均计算={平均('comp_delay_ms'):.3f} ms"
    )
    print(
        f"结构约束：平均冲突率={平均('conflict_rate') * 100:.2f}%，"
        f"平均负载不均衡={平均('load_imbalance'):.4f}"
    )
    print(
        f"切分行为：平均 l1={平均('mean_l1'):.3f}，"
        f"平均 l2={平均('mean_l2'):.3f}"
    )
    print(
        f"张量变化：平均上传={平均('mean_upload_bytes'):.1f} Bytes，"
        f"平均下行={平均('mean_downlink_bytes'):.1f} Bytes"
    )
    print(
        f"训练信号：平均 reward={平均('reward'):.6f}，"
        f"阶段末 epsilon={last_row['epsilon']:.4f}，"
        f"阶段末 ReplayBuffer={last_row['buffer_size']}"
    )
    print(
        f"阶段起止：第 {start_epoch} 轮完整时延="
        f"{first_row['full_total_delay_ms']:.3f} ms；"
        f"第 {epoch} 轮完整时延="
        f"{last_row['full_total_delay_ms']:.3f} ms"
    )

    # 用一句中文提示当前三个预设环境阶段之间的主要变化。
    if epoch == 50:
        print(
            "阶段变化提示：阶段一结束；下一阶段将进入车辆数量增加、"
            "可用 MIG 资源扩展的运行区间。"
        )
    elif epoch == 100:
        print(
            "阶段变化提示：阶段二结束；下一阶段将进入 20 MHz "
            "低带宽通信拥塞区间，应重点观察切分深度和通信时延变化。"
        )
    elif epoch == 150:
        print(
            "阶段变化提示：阶段三结束；本训练周期的三类动态 AI-RAN "
            "资源状态均已完成。"
        )

    print("#" * 88)
    print()


def train(
    *,
    episodes: int = 10,
    rounds: int = 150,
    seed: int = 42,
    checkpoint_path: str = "checkpoints/pcsfl_trained.pt",
    log_path: str = "logs/pcsfl/training_history.json",
    capacity_slack: int = 0,
    phase_balanced_replay: bool = True,
    use_joint_q: bool = True,
    reward_mode: str = "full",
    channel_mode: str = "state",
    device: str = "cpu",
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    provider = CIFAR10NonIIDProvider(num_clients=25, alpha=0.7, data_dir="./data")
    model = ResNet18_USFL(num_classes=10).to(torch.device(device))
    model.eval()
    tensor_cache = TensorProfileCache()
    agent = PCSFLAgent(
        state_dim=15,
        max_clients=20,
        max_migs=7,
        num_layers=7,
        hidden_dim=256,
        gamma=0.0,
        learning_rate=1e-4,
        replay_capacity=20_000,
        batch_size=48,
        target_sync_interval=50,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        capacity_slack=capacity_slack,
        phase_balanced_replay=phase_balanced_replay,
        use_joint_q=use_joint_q,
        device=device,
        seed=seed,
    )

    history = []
    global_step = 0
    random_channel_rng = np.random.default_rng(seed + 100_000)

    print("=" * 88)
    print("PCSFL 正式训练开始")
    print(
        f"训练配置：训练周期={episodes}，每周期轮数={rounds}，"
        f"初始随机种子={seed}，运行设备={device}"
    )
    print(
        f"模型配置：capacity_slack={capacity_slack}，"
        f"phase_balanced_replay={phase_balanced_replay}，"
        f"use_joint_q={use_joint_q}"
    )
    print(
        f"评价配置：reward_mode={reward_mode}，channel_mode={channel_mode}，"
        f"checkpoint={checkpoint_path}"
    )
    print("=" * 88)
    print()

    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env = LiquidAIRANEnv(data_provider=provider, max_vehicles=25)

        print()
        print("+" * 88)
        print(
            f"开始第 {episode:02d}/{episodes:02d} 个训练周期，"
            f"本周期随机种子={episode_seed}"
        )
        print("+" * 88)

        for epoch in range(1, rounds + 1):
            global_step += 1
            raw, migs, bandwidth, _ = env.step()
            state = build_agent_state(raw, migs, bandwidth)
            clusters, l1, l2, _ = agent.step(state, migs)
            metrics = evaluate_actions(
                raw_client_states=raw,
                available_migs=migs,
                current_bandwidth=bandwidth,
                cluster_choices=clusters,
                l1_choices=l1,
                l2_choices=l2,
                resnet=model,
                tensor_cache=tensor_cache,
                reward_mode=reward_mode,
                channel_mode=channel_mode,
                rng=random_channel_rng,
            )
            # The project is a one-step exogenous resource-allocation problem; next state does not bootstrap.
            loss = agent.observe(
                reward=metrics["reward"],
                next_state=state,
                next_available_migs=migs,
                done=True,
            )
            row = {
                "global_step": global_step,
                "episode": episode,
                "epoch": epoch,
                "seed": episode_seed,
                "n_clients": int(raw.shape[0]),
                "n_migs": int(migs),
                "bandwidth_mhz": float(bandwidth / 1e6),
                **metrics,
                "loss": loss,
                "epsilon": float(agent.epsilon),
                "buffer_size": len(agent.replay_buffer),
                "gradient_steps": int(agent.gradient_steps),
            }
            history.append(row)

            # 每一轮：简明中文输出。
            _打印每轮信息(
                episode=episode,
                episodes=episodes,
                epoch=epoch,
                rounds=rounds,
                row=row,
            )

            # 每十轮：打印更详细的系统、动作和学习状态。
            if epoch % 10 == 0:
                _打印十轮详细信息(
                    episode=episode,
                    epoch=epoch,
                    row=row,
                )

            # 每五十轮：统计当前训练周期最近 50 轮，形成阶段总结。
            if epoch % 50 == 0:
                _打印五十轮阶段总结(
                    episode=episode,
                    epoch=epoch,
                    history=history,
                )

        episode_rows = [row for row in history if row["episode"] == episode]
        if episode_rows:
            episode_full_mean = float(
                np.mean([row["full_total_delay_ms"] for row in episode_rows])
            )
            episode_conflict_mean = float(
                np.mean([row["conflict_rate"] for row in episode_rows])
            )
            episode_imbalance_mean = float(
                np.mean([row["load_imbalance"] for row in episode_rows])
            )
            print(
                f"第 {episode:02d} 个训练周期结束："
                f"平均完整U型时延={episode_full_mean:.3f} ms，"
                f"平均冲突率={episode_conflict_mean * 100:.2f}%，"
                f"平均负载不均衡={episode_imbalance_mean:.4f}，"
                f"当前 epsilon={agent.epsilon:.4f}"
            )

    agent.save(checkpoint_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 88)
    print("PCSFL 正式训练完成")
    print(f"模型检查点已保存：{checkpoint_path}")
    print(f"训练日志已保存：{log_path}")
    print(f"累计训练步数：{global_step}")
    print(f"累计梯度更新次数：{agent.gradient_steps}")
    print(f"最终 epsilon：{agent.epsilon:.4f}")
    print("=" * 88)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="训练最终 PCSFL 策略")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(episodes=args.episodes, rounds=args.rounds, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
