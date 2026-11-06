# PEMS04 代码阅读与本机运行记录

日期：2026-11-03（Asia/Shanghai）。源码起点：`13fb10c`。

## 程序如何工作

已阅读仓库全部 Python 源码、Shell 入口、测试、依赖声明和说明文档。

| 文件 / 模块 | 实际职责 |
| --- | --- |
| `download_pems.py`、`utils/pems.py` | PEMS04/07/08 下载、原始形状检查、图检查及 SHA-256 清单 |
| `prepare_pems04.py`、`utils/prepare_pems.py` | 生成滑动窗口、日历特征、邻接图及可选 UniCA 协变量 |
| `utils/data.py` | 数据验证、训练输入归一化、少样本抽样、同步打乱、尾批填充及张量转换 |
| `models/st_llm_tfp.py` | 时间投影、日历/节点嵌入、部分冻结 GPT-2、图注意力和预测头 |
| `models/timefilter_attention.py` | 动态相似度、物理/动态/剩余边分区、带训练噪声的 top-p 路由及 MoE 辅助损失 |
| `models/unica_adapter.py` | 静态/历史/已知未来协变量投影、变量选择、残差融合及未来注意力 |
| `ranger21.py` | 实际实现为 Ranger：RAdam、Lookahead、Gradient Centralization；不能仅凭文件名认定为 Ranger21 |
| `train_tfp.py` | 参数解析、数据加载、训练/验证、早停与最佳权重保存、权重重载及逐步预测评估 |
| `train_pems.sh` | TFP、UniCA 联合训练、仅适配器训练三种入口，默认六层 GPT-2、`U=1` |
| `utils/metrics.py` | PEMS 的正目标指标及 NYC 含零计数指标 |
| `tests/` | 离线单元测试、原始序列逐窗口核验、带梯度/形状/权重检查的真实训练入口 |
| `download_nyc_bike.py`、`prepare_nyc_bike.py`、`utils/nyc_bike.py`、`audit_nyc_bike.py`、`train_nyc_bike.sh` | NYC 小时计数聚合、按目标时间划分、天气与空间上下文处理及全量审计；本次未下载 NYC 数据 |

PEMS04 的张量路径：原始 `[16992,307,3]` → 窗口 `[B,12,307,3]` → 训练输入
`[B,3,307,12]` → 每节点 768 维 token `[B,307,768]` → 六层 GPT-2 → 预测
`[B,12,307,1]`。时间投影、日历嵌入、节点嵌入各提供 256 维，再经输入投影融合。

默认前五层使用因果注意力，最后一层使用 TimeFilter 图注意力，其偏置为
`[B,12,307,307]`。这里的因果顺序是**节点 token 顺序**，并非未来预测时间步。
前面各层仅训练 LayerNorm，最后 `U` 层还训练注意力参数（包括基础注意力参数和其中的
LoRA 参数）；所有层的 MLP 冻结，位置嵌入、最终 LayerNorm 和时空输入/输出模块可训练。

训练目标为原始流量单位的 MAE 加 `0.05 × MoE loss`。输入流量用训练窗口统计量标准化，
目标不标准化；模型预测在计算损失与指标之前反标准化。UniCA 默认在主干之前融合历史，
在主干之后融合已知未来日历；PEMS04 历史协变量为另外两个原始通道，静态协变量为节点
索引、度与加权度。`adapter_only` 从匹配的 TFP 权重初始化，仅更新适配器参数。

## 数据与评价协议

- 原始数据来自仓库下载器指定的 ASTGNN 镜像。下载哈希见 `data/PEMS04/download_manifest.json`。
- 输入 12 步，输出 12 步，一天 288 步，307 个节点。
- 先构造窗口，再连续划分 60/20/20：训练 10,181，验证 3,394，测试 3,394。
- 该既有协议的 split 边界附近共享部分原始时间戳；本次保留此划分。
- 原始边距离生成对称高斯加权图，并添加自环。
- 用验证 MAE 选择最佳权重，随后重载权重评估测试集。
- PEMS 仅评价目标大于零的位置。最终测试会裁掉尾批填充的重复样本。
- `test.csv` 前 12 行为各预测步，最后一行为它们的算术平均；RMSE 也是逐步 RMSE 的平均。
- MAPE、WMAPE 在 CSV 内是比例，转为百分数需乘 100。

## 环境与兼容性修复

所有安装与运行均使用 WSL Ubuntu-24.04 内的 `conda activate dl`。
Python 3.12.16、PyTorch 2.8.0+cu128、NumPy 2.5.4、pandas 3.1.0、
Transformers 5.22.1、PEFT 0.20.0；GPU 为 NVIDIA GeForce RTX 4080 SUPER，32 GiB。
完整依赖快照见 `ccfa-workfiles/checks/pems04/environment.txt`。

GPT-2 revision：`607a30d783dfa663caf39e06633721c8d4cfcd7e`。
本机默认 Xet 下载接口返回 404，使用 `HF_HUB_DISABLE_XET=1` 下载公开权重；完成缓存后
训练使用离线模式。缓存保存在用户的 Hugging Face 缓存目录。

本次修复 `TimeFilteredPFA.forward`：显式向前面的 GPT-2 block 传入因果掩码。
当前 Transformers 版本在 `GPT2Model` 外层准备该掩码，而本项目直接调用 block，原实现
传入 `None` 会失去 README 中声明的因果约束。最后 `U` 层仍按原图注意力路径计算。
新增离线回归测试通过扰动后面的节点 token 检查前面节点的中间输出：原始源码失败，
修复后通过。修复前的失败证据见 `ccfa-workfiles/checks/pems04/causal_regression_before_fix.log`。

## 可重复运行的命令

从 WSL 中执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd ~/Projects/TrafficFlowPrediction
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p ccfa-workfiles/checks/pems04

python -B download_pems.py --datasets pems04
python -B prepare_pems04.py
python -B tests/verify_prepared_data.py --data pems04 \
  --report ccfa-workfiles/checks/pems04/prepared_data_checks.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B -m unittest discover -s tests -v

# 首次运行需要下载 GPT-2；已有缓存时跳过这一步。
HF_HUB_DISABLE_XET=1 python -c 'from huggingface_hub import snapshot_download; snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"])'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 六层主模型：完整训练集一轮、完整验证集、完整测试集，无 batch 截断。
python -B tests/smoke_training.py --data pems04 --device cuda:0 \
  --llm_layer 6 --U 1 --batch_size 32 --epochs 1 \
  --save ccfa-workfiles/checks/pems04/tfp-full-

# 两种 UniCA 模式：检查真实数据路径，每种仅执行两次训练更新和一个评估 batch。
python -B tests/smoke_training.py --data pems04 --device cuda:0 \
  --llm_layer 6 --U 1 --batch_size 1 --epochs 1 --unica_enabled \
  --max_train_batches 2 --max_eval_batches 1 \
  --save ccfa-workfiles/checks/pems04/unica-smoke-
python -B tests/smoke_training.py --data pems04 --device cuda:0 \
  --llm_layer 6 --U 1 --batch_size 1 --epochs 1 --unica_enabled \
  --unica_training_policy adapter_only --tfp_moe_loss_weight 0 \
  --unica_pretrained_path ccfa-workfiles/checks/pems04/tfp-full-pems04/best_model.pth \
  --max_train_batches 2 --max_eval_batches 1 \
  --save ccfa-workfiles/checks/pems04/adapter-only-smoke-
```

## 实际执行结果

| 检查 | 结果 |
| --- | --- |
| Python 源码 AST 解析 | 27 个文件全部通过（含新增回归测试） |
| 两个 Shell 入口的语法检查 | 通过 |
| `python -m pip check` | 无依赖冲突 |
| 离线单元测试 | 22 / 22 通过 |
| 真实 PEMS04 全量数据核验 | 三个 split 的全部窗口、协变量、训练统计量和邻接图通过 |
| TFP 六层模型完整一轮 | 319 次优化器更新、107 个验证 batch、107 个测试 batch，共 533 次前向通过；梯度有限，权重重载成功 |
| TFP + UniCA 功能检查 | 2 次优化器更新、4 次前向通过；历史协变量维度正确识别为 2 |
| 仅适配器功能检查 | 同上；91 个适配器张量变化，所有主干张量逐项保持不变 |

TFP 完整一轮使用全部训练窗口，最终测试覆盖全部 **3,394** 个测试窗口，
没有使用 `max_train_batches` 或 `max_eval_batches`。总参数 84,153,918，
可训练参数 5,164,350。训练耗时 73.0329 秒，验证耗时 8.9488 秒；
峰值 PyTorch 分配显存 6.7928 GiB（不等于 `nvidia-smi` 的总占用）。
验证 MAE 为 29.9982，按验证集选中并重载第 1 轮权重。

| 测试范围 | MAE | RMSE | MAPE | WMAPE |
| --- | ---: | ---: | ---: | ---: |
| 预测第 1 步 | 24.2534 | 34.2055 | 27.6800% | 11.0317% |
| 预测第 12 步 | 33.2602 | 48.5419 | 31.1337% | 15.1218% |
| 12 个预测步平均 | **28.8236** | **41.8787** | **27.1854%** | **13.1067%** |

这是**训练 1 轮后的完整测试结果**，用于验证程序和真实数据全流程；未执行默认启动脚本的
200 轮训练，也没有据此声明复现收敛精度。两个 UniCA 检查各只使用 2 个训练 batch 和
1 个验证/测试 batch，其指标仅留在原始日志中，不用于模式间精度比较。

结果均保存在 `ccfa-workfiles/checks/pems04/`：

- `tfp-full-pems04/`：主模型权重 `best_model.pth`、逐步测试 `test.csv`、
  训练记录 `train.csv`、运行配置、抽样/归一化信息、选优信息及 `smoke_checks.json`。
- `unica-smoke-pems04/`、`adapter-only-smoke-pems04/`：两种适配器检查的相同产物。
- `tfp-full.log`、`unica-smoke.log`、`adapter-only-smoke.log`：原始训练与检查输出。
- `prepared_data_checks.json`、`unit_tests.log`、`environment.txt`、`gpt2_manifest.json`：
  数据核验、单元测试、环境及模型权重身份记录。

该工作目录已加入 `.gitignore`；原始数据仍位于 `data/PEMS04/`，准备结果位于
`data/pems04_12_12/`。如需按默认预算继续做新的完整训练，在同一 `dl` 环境运行
`bash train_pems.sh pems04 tfp`；该命令从 GPT-2 初始化开始新训练，不是恢复本次优化器状态。
