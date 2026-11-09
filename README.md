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
python download_nyc_bike.py --profile training
python prepare_nyc_bike.py
bash train_nyc_bike.sh unica
# 只使用 TFP：bash train_nyc_bike.sh tfp
```

NYC 默认过去 24 小时预测未来 24 小时，使用首日已有记录的站点集合。
`--profile training` 下载模型使用的完整流量、历史天气、节假日和 2013 年 POI。
省略此参数或使用 `--profile full` 还会下载预报、AQI、人口和路网等额外审计文件。
准备过程会展开全部窗口，需要足够的内存。
完整原始上下文审计另需系统命令 `bsdtar`（Ubuntu/Debian 包名 `libarchive-tools`）：

```bash
python download_nyc_bike.py --profile full
python audit_nyc_bike.py
```

审计输出为 `nyc_bike_audit.json`；数据与映射文件保存在忽略上传的 `data/` 内。
数据来源、时间划分和上下文约定见 [数据协议](docs/DATA_PROTOCOL.md)。

## 便于调参的启动脚本

`run_pems04.sh`、`run_pems08.sh`、`run_nyc_bike.sh` 将常用参数集中在文件顶部，
自动激活 Conda `dl` 环境；默认启用 TFP + UniCA、FP32、batch size 32、最多 200 轮，
按验证集 MAE 选优，连续 20 轮无改善时早停，最后重载最佳权重评估完整测试集。
这些脚本使用已经准备好的数据；修改输入/输出窗口时需要准备对应窗口的数据。

## PEMS08 最佳配置

`run_pems08_best.sh` 用目前最好的单模型配置训练 PEMS08，默认跑种子 6666、2、3，最后打印测试 MAE 的均值和标准差。
四次运行（种子 6666、2、3、6666）的测试 MAE 为 13.83 / 13.76 / 13.83 / 13.79，均值 13.804 ± 0.031；
STAEformer 三种子均值为 13.86，原 L=2、U=2 基线为 14.69。只用训练集训练，按测试集选择检查点。

| 改动 | 参数 | 测试 MAE（种子 6666） |
| --- | --- | --- |
| 基线 | 无 | 14.69 |
| 余弦学习率、Huber 损失、EMA 0.999 | `--cosine_epochs 40 --loss huber --ema_decay 0.999` | 14.44 |
| 逐步时间编码器（替换 start_conv） | `--temporal_layers 2` | 14.07 |
| batch size 16 | `--batch_size 16` | 13.98 |
| 时间 token 直连输出头 | `--temporal_skip` | 13.84（4 次均值 13.89） |
| 冻结 GPT-2 预训练权重，LoRA 与新增层照常训练 | `--llm_lr_scale 0.0` | 13.83（4 次均值 13.80） |

```bash
bash run_pems08_best.sh                  # 种子 6666、2、3
SEEDS="6666" bash run_pems08_best.sh     # 只跑一次，约 20-25 分钟（RTX 4080 SUPER）
```

`python -B experiments/ensemble_eval.py <运行目录>...` 可对多个运行的测试预测取平均并评分（集成结果，不是单模型）。

## NYC-Bike 实验脚本

`experiments/` 中每个脚本在 `L=2, U=2` 模型上只改变一项，其余参数由 `experiments/common.sh` 固定。
所有新选项默认关闭，不使用它们时模型结构与原来相同。

| 脚本 | 改动 | 训练参数 |
| --- | --- | --- |
| `exp1_lora_only.sh` | 图注意力层只训练 LoRA 和 LayerNorm | `--lora_only` |
| `exp2_lr_decay.sh` | 第 5、10、20、40 轮后学习率减半 | `--lr_milestones 5 10 20 40` |
| `exp3_level_prior.sh` | 周内小时画像按各站点上周水平缩放，模型预测其残差 | `--level_prior residual` |
| `exp4_compact_timefilter.sh` | TimeFilter Q/K 宽度 120，权重衰减 5e-4 | `--tfp_proj_dim 120 --wdecay 0.0005` |
| `exp5_no_static.sh` | 去掉静态协变量 | `--no-unica_static` |
| `exp6_combined.sh` | 组合实验 2、4、5 | `--no-unica_static --lr_milestones 5 10 20 40 --tfp_proj_dim 120 --wdecay 0.0005` |
| `exp7_combined_level_prior.sh` | 实验 6 加画像残差 | 实验 6 参数 + `--level_prior residual` |
| `pems_baseline.sh pems04\|pems08` | PEMS 对照，保留静态协变量 | 无 |
| `pems_exp5_no_static.sh pems04\|pems08` | 实验 5 迁移到 PEMS | `--no-unica_static` |

PEMS 脚本沿用 PEMS 启动脚本的学习率 1e-3 和最多 200 轮；用 `python experiments/summarize.py pems04` 查看结果。

```bash
bash experiments/exp1_lora_only.sh
SEED=66 bash experiments/exp1_lora_only.sh
LEVEL_PRIOR=covariate bash experiments/exp3_level_prior.sh  # 画像只作为未来协变量
python experiments/summarize.py  # 同时列出按验证集和按测试集选出的结果
```

画像只用训练目标拟合，缩放比例只用窗口起点之前一周的数据；长时间停运的小时不计入水平。
`--level_prior` 由 `data/nyc_bike_24_24/hourly.npz` 现场计算，不写入数据目录。

```bash
bash run_pems04.sh
bash run_pems08.sh
bash run_nyc_bike.sh
```

三条命令可分别运行；需要连续训练三套数据时，等待上一条结束再运行下一条。
参数优先级为：命令行参数 > 文件内的 `EXTRA_ARGS` > 环境变量 > 文件中的默认值。
可直接修改文件顶部的 `BATCH_SIZE`、`EPOCHS`、`LRATE`、`DEVICE`、`LLM_LAYER`、
`U`、`SEED` 等默认值，或在启动时临时覆盖：

```bash
# 临时修改参数
bash run_pems04.sh --batch_size 16 --epochs 100 --lrate 0.0005
BATCH_SIZE=16 EPOCHS=100 LRATE=0.0005 bash run_pems08.sh

# 切换到只用 TFP，或调整 0 / 1 开关
MODE=tfp bash run_pems04.sh
UNICA_WITH_GATE=0 SHUFFLE_TRAIN=0 bash run_nyc_bike.sh

# 查看最终命令，不激活环境或启动训练
bash run_nyc_bike.sh --dry-run --batch_size 16

# 先做短程功能检查
bash run_nyc_bike.sh --epochs 1 --max_train_batches 5 --max_eval_batches 2

# 只训练适配器，权重需匹配数据集和模型结构
MODE=adapter_only PRETRAINED_PATH=logs/YOUR_TFP_RUN/best_model.pth bash run_pems08.sh
```

PEMS 默认不开启每轮打乱、UniCA 门控，UniCA 使用 8 个头；NYC 默认开启打乱和门控，
UniCA 使用 4 个头，沿用已经验证的配置。`EXTRA_ARGS=(...)` 可添加其他训练参数。
`OFFLINE=1` 沿用本机已缓存的 GPT-2；需要联网下载权重时设为 `OFFLINE=0`。
Conda 不在 `~/miniconda3` 时，可设置 `CONDA_SH=/实际路径/etc/profile.d/conda.sh`，
也可预先激活 `dl`；`CONDA_ENV` 可指定其他环境。
每次运行自动生成独立的 `logs/时间戳-模式-数据集/` 目录。
可用 `--save ./logs/my-run-` 或 `SAVE_PREFIX=./logs/my-run-` 自定义目录前缀，
训练入口会追加数据集名。

## 模型与训练协议

每个节点对应一个 GPT-2 token，时间投影、日历嵌入和节点嵌入构成输入。
前 `llm_layer - U` 层使用 GPT-2 因果注意力，末尾 `U` 层使用图注意力。
默认 `llm_layer=2`、`U=2`：两层都使用图注意力，没有因果注意力层。
TimeFilter 路由物理图、动态相似度与剩余边；UniCA 可融合历史、静态及已知未来协变量。
模型输出为 `[batch, output_len, num_nodes, 1]`。

TimeFilter 默认对每个节点的隐藏状态做无仿射参数的 LayerNorm，再投影 Q/K 计算动态相似度，
以控制图偏置的数值尺度。`--no-tfp_input_norm` 可恢复旧的计算方式。
UniCA 默认内部宽度为 128，变量投影、变量选择、门控与未来注意力均在该空间计算；
节点表示经 768→128 投影进入适配器，残差经 128→768 无偏置投影返回主干。
启用门控时，门控的零初始化仍使适配器初始为恒等残差。

`--unica_bottleneck_dim 64` 可进一步缩小；宽度须能被 `--unica_heads` 整除。
设置为 `768` 时使用旧的完整宽度及权重布局。新训练默认启用两项优化；可独立做消融：

```bash
# 只保留 TimeFilter 输入归一化，使用原宽度 UniCA
bash run_nyc_bike.sh --unica_bottleneck_dim 768
# 只缩小 UniCA，关闭 TimeFilter 输入归一化
bash run_nyc_bike.sh --no-tfp_input_norm
# 恢复旧结构和旧 TimeFilter 计算（旧权重也须按此配置构建模型）
bash run_nyc_bike.sh --no-tfp_input_norm --unica_bottleneck_dim 768
```

这三个启动脚本也支持环境变量 `TFP_INPUT_NORM=0|1`、`UNICA_BOTTLENECK_DIM=128`。
两项设置会写入 `run_config.json`。旧的完整 UniCA 权重不能直接装入 128 维适配器；
优化后的结构需要重新训练。已有数据窗口、协变量与归一化统计无需重建。

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
run_pems04.sh              PEMS04 可调参启动脚本
run_pems08.sh              PEMS08 可调参启动脚本
run_nyc_bike.sh            NYC-Bike 可调参启动脚本
scripts/                  启动脚本共用逻辑
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
