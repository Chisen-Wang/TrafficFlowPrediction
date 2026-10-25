# TFP: ST-LLM + TimeFilter + UniCA

Traffic forecasting with graph-aware GPT-2 attention, TimeFilter routing, and optional UniCA covariate fusion.

这是 TFP 的精简研究代码，支持 PEMS04、PEMS07、PEMS08 和 NYC-Bike。
保留模型、数据准备、统一训练入口与测试；不包含基线对比模块、原始数据、预训练权重或历史实验结果。

## 安装

使用 Python **3.11+**，本次本地检查使用 Python 3.12。以下命令均从仓库根目录运行。
正式训练建议使用 CUDA GPU；小规模功能检查可以使用 `--device cpu`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需特定 CUDA 构建，请先安装与机器匹配的 PyTorch，再安装上述依赖。
依赖范围不代表每个版本组合都已验证；本地验证环境见 [验证记录](docs/VALIDATION.md)。
首次构建模型会下载 Hugging Face 的 `gpt2`；离线运行前需准备相应缓存。

## PEMS 快速开始

以 PEMS08 为例：

```bash
python download_pems.py --datasets pems08
python prepare_pems08.py
bash train_pems.sh pems08 tfp
```

PEMS04/07 将上述数据集名和准备脚本中的数字相应替换即可。
下载脚本支持 `--datasets pems04 pems07 pems08`，并记录下载文件的 SHA-256。
也可以自行准备原始数据，使用 `--raw-data` 和 `--distance-csv` 指定位置。

| 数据集 | 节点数 | 默认原始文件目录 | 默认准备结果 | 默认窗口 |
| --- | ---: | --- | --- | --- |
| PEMS04 | 307 | `data/PEMS04/` | `data/pems04_12_12/` | 12 → 12 |
| PEMS07 | 883 | `data/PEMS07/` | `data/pems07_12_12/` | 12 → 12 |
| PEMS08 | 170 | `data/PEMS08/` | `data/pems08_12_12/` | 12 → 12 |

原始文件名为 `PEMSxx.npz`、`PEMSxx.csv`。准备结果包含 `train.npz`、`val.npz`、
`test.npz`、`adj_mx.pkl`、`metadata.json` 及 UniCA 协变量。
PEMS07 节点较多，启动脚本默认 batch size 为 4，其余 PEMS 数据集为 32；显存不足时继续调低。
准备过程会展开全部窗口，需要足够的内存和磁盘空间。

启用 UniCA，或使用已有 TFP 权重只训练适配器：

```bash
bash train_pems.sh pems08 unica
bash train_pems.sh pems08 adapter_only \
  --unica_pretrained_path logs/YOUR_TFP_RUN/best_model.pth
```

`YOUR_TFP_RUN` 需替换成实际训练目录。适配器权重必须与数据集、邻接图、窗口长度和模型结构匹配。
启动脚本在前台运行，末尾可追加训练参数，例如：

```bash
bash train_pems.sh pems08 tfp \
  --batch_size 1 --epochs 1 --max_train_batches 2 --max_eval_batches 1
```

以上短程运行只检查功能，不作为模型性能结果。

自定义窗口时，准备和训练参数需保持一致：

```bash
python prepare_pems04.py --input-len 96 --output-len 12
python train_tfp.py --data pems04 --input_len 96 --output_len 12
```

少样本实验使用统一入口，目前仅支持不启用 UniCA 的 TFP：

```bash
python train_tfp.py --data pems04 --train_fraction 0.1 --subset_seed 6666
```

## NYC-Bike

```bash
python -m pip install -r requirements-nyc-bike.txt
python download_nyc_bike.py
python prepare_nyc_bike.py
bash train_nyc_bike.sh unica
# 只使用 TFP：bash train_nyc_bike.sh tfp
```

NYC 默认过去 24 小时预测未来 24 小时，使用首日已有记录的站点集合。
下载脚本包含天气、POI、人口等上下文文件，下载量和准备内存需求较大。
完整原始上下文审计另需系统命令 `bsdtar`（Ubuntu/Debian 包名 `libarchive-tools`）：

```bash
python audit_nyc_bike.py
```

审计输出为 `nyc_bike_audit.json`；数据与映射文件保存在忽略上传的 `data/` 内。
数据来源、时间划分和上下文约定见 [数据协议](docs/DATA_PROTOCOL.md)。

## 模型与训练协议

每个节点对应一个 GPT-2 token，时间投影、日历嵌入和节点嵌入构成输入。
前 `llm_layer - U` 层使用 GPT-2 因果注意力，末尾 `U` 层使用图注意力。
TimeFilter 路由物理图、动态相似度与剩余边；UniCA 可融合历史、静态及已知未来协变量。
模型输出为 `[batch, output_len, num_nodes, 1]`。

- 默认使用验证集 MAE 选优和早停，训练结束重载最佳权重后评估测试集。
- 保留 `--model_selection_split train|val|test` 供显式实验使用；选择 `test` 会让测试集参与模型选择，不能当作独立测试结果。
- PEMS 沿用“先构造窗口，再按时间划分 60/20/20”的协议，划分边界附近的原始时间范围存在重叠。NYC 按目标时间划分，跨 split 的目标时间集合不相交。
- 仅以训练输入流量计算归一化统计量，目标保留原始单位；少样本时仅使用被选中的训练窗口。
- PEMS 指标只评价正目标，尾批沿用末样本填充。NYC 的 MAE/RMSE/WMAPE 包含零需求，MAPE 只用正目标，尾批不填充。
- MAPE、WMAPE 输出为比值；显示百分数时需乘以 100。

默认训练产物位于 `logs/`，包括 `best_model.pth`、`run_config.json`、
`sampling.json`、`train_indices.npy`、`train.csv`、`test.csv` 和 `model_selection.json`。
`--save` 接受目录**前缀**，训练入口会在其末尾追加数据集名。
完整参数见 `python train_tfp.py --help`。

## 测试

无需真实数据、GPU 或 GPT-2 权重的单元测试：

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest discover -s tests -v
```

准备好三套 PEMS 的 12→12 数据、GPT-2 缓存和 CUDA 后，可运行额外集成检查：

```bash
python -B tests/verify_prepared_data.py
python -B tests/run_smoke_suite.py
```

GPU 集成脚本会执行三数据集 × 三训练模式，共 9 次短程运行，并将输出放入 `logs/`。
GitHub Actions 仅运行不依赖外部数据和模型权重的单元测试及 Shell 语法检查。

## 目录

```text
models/                    TFP、TimeFilter、UniCA
utils/                     加载、预处理和评价指标
tests/                     单元测试、完整数据核验、集成检查
docs/                      数据协议、整理说明和验证记录
train_tfp.py               统一训练入口
train_pems.sh              PEMS 三种训练模式
train_nyc_bike.sh          NYC-Bike 启动脚本
prepare_*.py               数据准备入口
download_*.py              数据下载工具
audit_nyc_bike.py          NYC 原始上下文和准备结果审计
ranger21.py                既有 Ranger 优化器（文件名沿用原项目）
```

## 发布与许可状态

本次按项目维护者要求**暂不指定项目许可证**，没有添加 `LICENSE`。
此目录是可上传的源码整理包，不声明所有内容已经完成开源授权核验。
来源与许可记录见 [第三方说明](THIRD_PARTY_NOTICES.md)，整理范围见 [整理说明](docs/RELEASE_NOTES.md)。
数据与预训练权重单独获取，不随本仓库发布。
