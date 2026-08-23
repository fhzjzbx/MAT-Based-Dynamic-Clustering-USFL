# PCSFL 项目 README

> **Parallel Clustering–Splitting Federated Learning for Dynamic U-Shaped AI-RAN**
>
> 本目录包含本项目最终采用的 PCSFL 实现。代码已经去除 `v1`、`v7`、`coord`、`coord_learn_v2` 等开发阶段版本号，文件名只表示其在项目中的职责，便于后续提交、答辩、复现实验和继续开发。

---

## 1. 一句话看懂这个项目

本项目面向动态 6G AI-RAN 场景，在每一轮根据当前客户端状态、无线信道、可用 MIG 数量和系统带宽，同时决定：

1. **每个客户端分配到哪个 MIG；**
2. **每个 MIG 使用哪一组 U 型切分点 `(l1, l2)`；**
3. 在满足 MIG 容量约束的前提下，使完整 U 型 Split Federated Learning 的系统时延尽可能小。

最终 PCSFL 将早期“客户端独立选择切分点”的方式改为：

```text
客户端级决策：Client → MIG
MIG 级决策：MIG → (l1, l2)
```

因此，同一个 MIG 内的客户端天然使用一致的切分点，避免批处理阶段出现特征张量维度冲突。

---

# 2. 最终代码结构

将本代码放入原项目根目录后，建议目录保持如下结构：

```text
PCSFL/
│
├── baselines/
│   └── pcsfl/
│       ├── __init__.py
│       ├── pcsfl_agent.py          # PCSFL 决策网络、动作生成、ReplayBuffer、Joint-Q 学习
│       ├── pcsfl_state.py          # 将环境原始状态构造成 PCSFL 使用的 15 维状态
│       └── pcsfl_physics.py        # U 型切分物理时延、张量大小和 reward 计算
│
├── train_pcsfl.py                  # 正式训练入口
├── evaluate_pcsfl.py               # 单个 seed 的训练/未训练模型评估
├── evaluate_pcsfl_multiseed.py     # 5-seed same-trace 公平稳定性评估
├── compare_pcsfl.py                # 对比 trained 与 untrained 的核心指标
├── pcsfl_evaluation.py             # 公共评估、轨迹生成和统计函数
├── test_pcsfl.py                   # 算法 Smoke Test
├── check_pcsfl_suite.py            # 文件完整性和 Python 语法检查
│
├── experiments/
│   ├── train_ablation.py           # 消融模型训练
│   ├── evaluate_ablation_multiseed.py
│   │                               # 5-seed 消融公平评估
│   ├── plot_results.py             # 自动生成最终实验图
│   └── make_tables.py              # 自动生成 CSV/JSON 实验表
│
├── checkpoints/                    # 训练后自动创建
├── logs/                           # 训练和评估结果
├── figures/                        # 绘图结果
└── tables/                         # 最终实验表格
```

原项目中的以下两个核心文件保持原样：

```text
envs/liquid_airan_env.py
models/usfl_networks.py
```

PCSFL 通过现有接口读取环境状态并调用 U-SFL 模型，不需要改动原环境和原模型主体。

---

# 3. 整个 PCSFL 数据流

下面这张结构图基本可以概括全部代码。

```text
┌──────────────────────────────────────────────┐
│              LiquidAIRANEnv                  │
│                                             │
│  输出：                                     │
│  - 当前客户端状态                           │
│  - available_migs                           │
│  - current_bandwidth                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              raw_client_states
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              pcsfl_state.py                  │
│                                             │
│  local state                                │
│  - channel                                  │
│  - compute                                  │
│  - CIFAR-10 label distribution              │
│                                             │
│  + global context                           │
│  - bandwidth                                │
│  - available MIGs                           │
│  - active clients                           │
│                                             │
│              → 15 维状态                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              PCSFLAgent                     │
│              pcsfl_agent.py                 │
│                                             │
│   Shared MLP backbone                       │
│        │                    │               │
│        ▼                    ▼               │
│ Client → MIG Q         MIG → Split Q        │
│                            │                │
│                            ▼                │
│                       (l1, l2)              │
│                                             │
│ + exact capacity constraint                 │
│ + MIG-level unified split                   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
       cluster_choices / l1_choices / l2_choices
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             pcsfl_physics.py                │
│                                             │
│ Client Part A                               │
│      │                                      │
│      └── l1 → upload tensor size            │
│                                             │
│ MIG Part B                                  │
│      │                                      │
│      └── l2 → downlink tensor size          │
│                                             │
│ 使用同一轮 channel 计算通信时延             │
│ + MIG logical compute delay                 │
│                                             │
│ → full U-shaped system delay                │
│ → reward                                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              PCSFL learning                 │
│                                             │
│ Joint-Q system value                        │
│ + Phase-balanced Replay                     │
│ + target network                            │
│ + ε-greedy exploration                      │
└──────────────────────────────────────────────┘
```

---

# 4. 三个最重要的核心文件

## 4.1 `baselines/pcsfl/pcsfl_agent.py`

这是整个 PCSFL 的核心算法文件。

它主要负责四件事：

```text
1. 根据 15 维系统状态输出动作；
2. 决定 Client → MIG；
3. 决定每个 MIG 的统一 (l1, l2)；
4. 使用 ReplayBuffer 和 Joint-Q 更新策略。
```

### 动作输出接口

Agent 最终仍遵守项目规定的接口：

```python
cluster_choices, l1_choices, l2_choices, bw_weights = agent.step(
    active_clients_state,
    available_migs,
)
```

返回数组长度始终等于当前真实客户端数量 `N`：

```text
cluster_choices : 每个客户端所属 MIG
l1_choices      : 每个客户端第一切分点
l2_choices      : 每个客户端第二切分点
bw_weights      : PCSFL 中固定为 1
```

切分点始终满足：

```text
0 <= l1 <= l2 < 7
```

### 为什么不是每个客户端独立选择 `(l1,l2)`

早期实现中可能出现：

```text
MIG 0
├── Client 0 → (1, 4)
├── Client 1 → (3, 5)
└── Client 2 → (2, 6)
```

同一 MIG 中切分位置不同后，中间特征张量的维度可能不一致，无法直接组成 batch。

最终结构改成：

```text
MIG 0 → 统一选择 (3, 5)

MIG 0
├── Client 0 → (3, 5)
├── Client 1 → (3, 5)
└── Client 2 → (3, 5)
```

因此切分一致性直接由动作结构保证。

### Exact Capacity Constraint

默认配置：

```python
capacity_slack = 0
```

它限制各 MIG 的设备数量尽量均衡。

例如：

```text
10 clients / 2 MIG
→ [5, 5]

11 clients / 2 MIG
→ [5, 6]

16 clients / 5 MIG
→ [3, 3, 3, 3, 4]
```

这里限制的是**每个 MIG 最终承载多少客户端**。

具体哪一个客户端进入哪一个 MIG，仍由 PCSFL 根据客户端状态学习决定。

---

## 4.2 `baselines/pcsfl/pcsfl_state.py`

这个文件只负责**状态构造**。

环境原始客户端状态为：

```text
12 维：
[channel, compute, 10-dimensional label distribution]
```

PCSFL 再增加三个全局系统变量：

```text
bandwidth
available_migs
active_clients
```

最终得到：

```text
15-dimensional PCSFL state
```

状态组成可以写成：

```text
s_n =
[
    channel_n,
    compute_n,
    label_distribution_n,
    bandwidth,
    available_migs,
    n_clients
]
```

最终实现还保留绝对信道质量，使 Agent 能够识别“整体信道较好”和“整体信道较差”的轮次。

---

## 4.3 `baselines/pcsfl/pcsfl_physics.py`

这个文件负责将 Agent 动作转换为真实系统指标。

完整 U 型 SFL 路径为：

```text
Client
  │
  │ Part A
  ▼
l1 smashed tensor
  │
  │ uplink
  ▼
MIG
  │
  │ Part B
  ▼
l2 tensor
  │
  │ downlink
  ▼
Client
  │
  │ Part C
  ▼
output
```

因此：

```text
l1 主要决定 uplink tensor size
l2 主要决定 downlink tensor size
```

最终评价指标使用：

```text
full U-shaped delay
=
uplink delay
+
logical compute delay
+
downlink delay
```

为了保留与早期实验的兼容性，结果中同时保留：

```text
mean_total_delay_ms
```

表示旧口径：

```text
uplink + logical compute
```

以及：

```text
mean_full_total_delay_ms
```

表示最终正式指标：

```text
uplink + logical compute + downlink
```

论文、报告和最终实验应优先使用：

```text
mean_full_total_delay_ms
```

---

# 5. 其他文件分别做什么

| 文件 | 作用 | 正常使用频率 |
|---|---|---|
| `train_pcsfl.py` | 正式训练 PCSFL | 必须 |
| `evaluate_pcsfl.py` | 单个 seed 正式评估 | 必须 |
| `evaluate_pcsfl_multiseed.py` | 5 个 seed 公平稳定性评估 | 最终实验必须 |
| `compare_pcsfl.py` | 快速比较训练前后结果 | 推荐 |
| `pcsfl_evaluation.py` | 评估公共函数，被其他脚本调用 | 不直接运行 |
| `test_pcsfl.py` | 测试动作合法性、容量、冲突和学习路径 | 第一次部署必须 |
| `check_pcsfl_suite.py` | 检查文件和 Python 语法 | 第一次部署必须 |
| `experiments/train_ablation.py` | 训练消融版本 | 做论文实验时使用 |
| `experiments/evaluate_ablation_multiseed.py` | 公平比较消融版本 | 做论文实验时使用 |
| `experiments/plot_results.py` | 根据 JSON 自动画图 | 最终报告使用 |
| `experiments/make_tables.py` | 自动生成实验表格 | 最终报告使用 |

---

# 6. 环境要求

PCSFL 依赖原项目环境，因此需要确保原工程已经能够正常导入：

```python
from data.cifar_10_provider import CIFAR10NonIIDProvider
from envs.liquid_airan_env import LiquidAIRANEnv
from models.usfl_networks import ResNet18_USFL
from interfaces.base_agent import BaseAgent
```

同时需要：

```text
Python
NumPy
PyTorch
CIFAR-10 数据
```

具体 Python/PyTorch 版本以原项目可以正常运行的环境为准，本 PCSFL 包不强制重新建立独立环境。

如果项目目录为：

```text
D:\PCSFL
```

应确保：

```text
D:\PCSFL\data\
D:\PCSFL\envs\
D:\PCSFL\models\
D:\PCSFL\interfaces\
D:\PCSFL\baselines\
```

均存在。

---

# 7. 第一次拿到代码应该怎么运行

推荐严格按照以下顺序。

## Step 1：进入项目目录

Windows CMD：

```bat
cd /d D:\PCSFL
```

## Step 2：检查代码完整性

```bat
python check_pcsfl_suite.py
```

正常应看到：

```text
PCSFL SUITE CHECK PASSED
```

如果这一步失败，先处理缺失文件或 Python 语法错误，不要直接训练。

## Step 3：运行 Smoke Test

```bat
python test_pcsfl.py
```

测试内容主要包括：

```text
动态客户端数量
动态 MIG 数量
合法 cluster action
0 <= l1 <= l2 < 7
MIG 内统一切分
exact capacity constraint
Joint-Q 更新
U-shaped tensor evaluator
```

Smoke Test 成功后再进入正式实验。

---

# 8. 先测试未训练 PCSFL

在训练之前先建立结构基线：

```bat
python evaluate_pcsfl.py --untrained
```

默认使用：

```text
seed = 2026
150 rounds
```

输出目录：

```text
logs/pcsfl/evaluation/
```

主要文件：

```text
untrained_summary.json
untrained_detail.json
untrained_detail.csv
```

这里的结果回答：

> 在相同结构下，不经过学习时 PCSFL 的性能是多少？

---

# 9. 正式训练 PCSFL

默认运行：

```bat
python train_pcsfl.py
```

默认训练量：

```text
10 episodes × 150 rounds
=
1500 training steps
```

默认设备：

```text
CPU
```

输出模型：

```text
checkpoints/pcsfl_trained.pt
```

输出训练日志：

```text
logs/pcsfl/training_history.json
```

## 自定义训练参数

例如：

```bat
python train_pcsfl.py --episodes 10 --rounds 150 --seed 42 --device cpu
```

如果环境支持 CUDA，可以尝试：

```bat
python train_pcsfl.py --device cuda
```

建议第一次运行仍使用：

```text
cpu
```

确认整个项目流程稳定后再切换设备。

---

# 10. 正式评估训练模型

训练完成后：

```bat
python evaluate_pcsfl.py
```

默认加载：

```text
checkpoints/pcsfl_trained.pt
```

结果输出：

```text
logs/pcsfl/evaluation/
├── trained_summary.json
├── trained_detail.json
└── trained_detail.csv
```

---

# 11. 一条命令比较训练前后

必须先运行：

```bat
python evaluate_pcsfl.py --untrained
python evaluate_pcsfl.py
```

然后：

```bat
python compare_pcsfl.py
```

输出类似：

```text
PCSFL: trained vs untrained

legacy:
85.xxx -> 80.xxx ms

full:
90.xxx -> 84.xxx ms

conflict rate:
0.000%

load imbalance:
0.06xx
```

重点查看：

```text
full
```

即完整 U 型时延。

---

# 12. 最正式的实验：5-seed same-trace 公平评估

运行：

```bat
python evaluate_pcsfl_multiseed.py
```

默认 seed：

```text
2026
2027
2028
2029
2030
```

公平性设计如下：

```text
seed 2026
   │
   ▼
先完整生成一份 150-round environment trace
   │
   ├───────────────┐
   ▼               ▼
Untrained       Trained
   │               │
   └───────┬───────┘
           ▼
      公平比较
```

也就是说，同一个 seed 下，两种模型经历完全一致的：

```text
客户端数量
MIG 数量
带宽
channel
compute
label distribution
```

输出：

```text
logs/pcsfl/multiseed/
├── multiseed_summary.json
└── multiseed_results.csv
```

## 最终主要看哪些数字

打开：

```text
logs/pcsfl/multiseed/multiseed_summary.json
```

重点看：

```text
aggregate
```

其中最重要的是：

```text
untrained_full_mean_ms
trained_full_mean_ms
gain_mean_pct
gain_std_pct
wins
phase1_gain_mean_pct
phase2_gain_mean_pct
phase3_gain_mean_pct
```

历史冻结实验的参考结果为：

```text
Untrained full:
89.613 ± 0.823 ms

Trained full:
84.857 ± 1.310 ms

Mean gain:
5.31 ± 0.85%

Trained wins:
5 / 5
```

如果使用当前职责化代码重新训练，由于随机初始化、训练轨迹或实现整理可能产生小幅数值差异，应以重新运行得到的 JSON 为准。

---

# 13. 三个动态阶段怎么看

150 轮环境分为三个典型阶段：

```text
Phase 1
round 1–50
100 MHz / 2 MIG

Phase 2
round 51–100
100 MHz / 5 MIG

Phase 3
round 101–150
20 MHz / 2 MIG
```

Phase 3 是最明显的通信拥塞阶段。

最终 PCSFL 希望学到的行为为：

```text
Bandwidth 降低
      ↓
Agent 感知 20 MHz
      ↓
选择更深的 l1 / 合适的 l2
      ↓
上传和下行 Tensor 变小
      ↓
Communication Delay 降低
      ↓
Full U-shaped Delay 降低
```

因此最终实验不应只看总体平均时延，还需要同时检查：

```text
mean_l1
mean_l2
mean_upload_bytes
mean_downlink_bytes
Phase 3 full delay
```

---

# 14. 消融实验怎么运行

消融实验全部放在：

```text
experiments/
```

避免影响正式 PCSFL 代码。

## Step 1：训练全部消融模型

```bat
python experiments\train_ablation.py --variant all
```

主要消融包括：

```text
no_phase_balance
legacy_reward
random_channel_reward
slack1
separate_q
```

## Step 2：进行 5-seed 公平消融评估

```bat
python experiments\evaluate_ablation_multiseed.py
```

结果：

```text
logs/pcsfl_ablation/multiseed/
├── ablation_summary.json
└── ablation_results.csv
```

重点指标：

```text
delta_vs_full_mean_ms
```

解释：

```text
delta > 0
→ 去掉/削弱该组件后更慢
→ Full PCSFL 中该组件有正贡献

delta ≈ 0
→ 独立影响较小

delta < 0
→ 消融版本反而更快
→ 不能宣称原组件有明确性能贡献
```

---

# 15. 自动生成实验图

完成 multi-seed 和 ablation 后：

```bat
python experiments\plot_results.py
```

图片输出：

```text
figures/pcsfl/
```

主要包括：

```text
01_multiseed_full_delay.png
02_multiseed_training_gain.png
03_phase_training_gain.png
04_phase_full_delay.png
05_ablation_full_delay.png
06_ablation_delta_vs_full.png
07_training_full_delay_ma.png
```

---

# 16. 自动生成实验表

运行：

```bat
python experiments\make_tables.py
```

输出：

```text
tables/pcsfl/
├── multiseed.csv
├── ablation.csv
└── key_metrics.json
```

这些文件可以直接用于论文或实验报告的数据整理。

---

# 17. 最重要的输出目录

如果只想知道结果在哪里，看这一节即可。

```text
checkpoints/
└── pcsfl_trained.pt
        ↑
        最终训练模型
```

```text
logs/pcsfl/training_history.json
        ↑
        训练过程
```

```text
logs/pcsfl/evaluation/
        ↑
        单 seed 训练前/后结果
```

```text
logs/pcsfl/multiseed/
        ↑
        最正式的 5-seed 公平测试
```

```text
logs/pcsfl_ablation/
        ↑
        消融实验
```

```text
figures/pcsfl/
        ↑
        最终实验图片
```

```text
tables/pcsfl/
        ↑
        最终实验表
```

---

# 18. 结果字段是什么意思

| 字段 | 含义 |
|---|---|
| `mean_total_delay_ms` | 旧实验兼容口径：uplink + compute |
| `mean_full_total_delay_ms` | 最终正式指标：uplink + compute + downlink |
| `mean_tx_delay_ms` | 通信相关时延统计 |
| `mean_uplink_delay_ms` | Client → MIG 时延 |
| `mean_downlink_delay_ms` | MIG → Client 时延 |
| `mean_comp_delay_ms` | 逻辑 MIG 计算/排队时延 |
| `mean_conflict_rate` | MIG 内切分冲突率，最终结构应为 0 |
| `mean_load_imbalance` | MIG 负载不均衡程度，越低越好 |
| `mean_l1` | 平均第一切分深度 |
| `mean_l2` | 平均第二切分深度 |
| `mean_upload_bytes` | 平均上传 Tensor 大小 |
| `mean_downlink_bytes` | 平均下行 Tensor 大小 |

最终报告中优先采用：

```text
mean_full_total_delay_ms
```

---

# 19. 最终算法为什么比早期版本更合理

早期 PCSFL 主要采用：

```text
每个 Client 独立输出：
cluster + l1 + l2
```

即使不断修改 reward，也很难彻底解决：

```text
同一 MIG 内切分不一致
        ↓
Tensor dimension conflict
        ↓
无法正常 batch
        ↓
退化成逐客户端处理
        ↓
计算时延显著增加
```

最终 PCSFL 把问题改造成：

```text
Client → MIG
+
MIG → unified (l1,l2)
+
exact capacity
+
full U-shaped physical objective
+
Joint-Q
+
phase-balanced replay
```

因此优化重点已经从“继续修改奖励系数”转向“使决策结构与 U 型 SFL 的真实物理约束一致”。

---

# 20. 关于 `gamma=0.0`

正式训练中：

```python
agent.observe(
    reward=...,
    next_state=...,
    done=True,
)
```

每一轮都按照 one-step resource allocation transition 处理，因此 TD target 不进行下一状态 bootstrap。

当前实现设置：

```python
gamma = 0.0
```

与这种 one-step 训练语义保持一致。

在项目报告中，不应将 `gamma=0` 单独描述为已经通过消融实验验证的性能贡献。

---

# 21. 关于旧 checkpoint

职责化版本默认使用：

```text
checkpoints/pcsfl_trained.pt
```

如果之前保存有旧名称 checkpoint，请先保留原文件。

不要仅通过修改文件名就假定旧 checkpoint 与当前整理后的网络结构完全兼容。

推荐：

```text
旧代码 + 旧 checkpoint
→ 保留，用于复现历史实验

当前职责化代码
→ 重新训练
→ checkpoints/pcsfl_trained.pt
→ 作为正式工程版本
```

---

# 22. 最常用命令速查

第一次检查：

```bat
python check_pcsfl_suite.py
python test_pcsfl.py
```

未训练基线：

```bat
python evaluate_pcsfl.py --untrained
```

训练：

```bat
python train_pcsfl.py
```

训练模型评估：

```bat
python evaluate_pcsfl.py
```

训练前后比较：

```bat
python compare_pcsfl.py
```

5-seed 正式公平测试：

```bat
python evaluate_pcsfl_multiseed.py
```

全部消融：

```bat
python experiments\train_ablation.py --variant all
python experiments\evaluate_ablation_multiseed.py
```

画图：

```bat
python experiments\plot_results.py
```

生成表格：

```bat
python experiments\make_tables.py
```

---

# 23. 如果只记住五件事

```text
1. pcsfl_agent.py
   = 决策和学习核心

2. pcsfl_state.py
   = 环境状态 → 15 维 PCSFL 状态

3. pcsfl_physics.py
   = 动作 → U 型物理时延 / reward

4. train_pcsfl.py
   = 正式训练

5. evaluate_pcsfl_multiseed.py
   = 最终最可信的公平实验
```

整个最终 PCSFL 可以概括为：

> **基于动态 AI-RAN 状态进行客户端到 MIG 的联合聚类，并在 MIG 级统一选择 U 型切分点；通过精确容量约束、完整上下行物理时延建模、Joint-Q 系统价值学习和分阶段经验采样，使决策结构与真实 U-shaped SFL 执行约束保持一致。**
