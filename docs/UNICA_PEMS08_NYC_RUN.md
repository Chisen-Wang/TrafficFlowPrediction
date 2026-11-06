# PEMS08 与 STContext NYC-Bike：UniCA 运行记录

日期：2026-11-03（Asia/Shanghai）。环境：WSL Ubuntu-24.04，`conda activate dl`。
本次验证范围为每套真实数据训练一轮，并评估完整验证集、重载最佳权重、评估完整测试集。
两套运行都启用 TFP 和 UniCA，使用 GPT-2 六层、`U=1`、FP32、Ranger、随机种子 6666，
按验证集 MAE 选择权重。没有设置训练或评估 batch 截断。

## 实际结果

两套运行均已成功结束，最佳权重重载后的完整测试结果如下：

| 数据集 | MAE | RMSE | MAPE | WMAPE | 峰值 PyTorch 分配显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS08，12 个预测步平均 | 22.1119 | 32.1992 | 19.1290% | 9.5254% | 6.4167 GiB |
| NYC-Bike，24 个预测步平均 | 1.9416 | 3.8469 | 61.6533% | 49.6194% | 2.9784 GiB |

| 执行证据 | PEMS08 | NYC-Bike |
| --- | ---: | ---: |
| 优化器更新次数 | 335 | 5473 |
| 验证 batch 数 | 112 | 2191 |
| 测试 batch 数 | 112 | 1627 |
| 总前向次数 | 559 | 9291 |
| 更新的 UniCA 张量数 | 91 | 123 |
| 验证 MAE | 22.8295 | 1.7512 |
| 训练耗时 | 72.9714 秒 | 520.8270 秒 |
| 验证耗时 | 10.3713 秒 | 55.2269 秒 |

所有前向输出及训练梯度均为有限值，预测和注意力形状检查通过，
两套测试 CSV 分别包含 12/24 个预测步和一行均值。
NYC 最后一个测试 batch 只有一个样本，保存的最后输出形状 `[1,24,326,1]` 符合不填充尾批的设置。
显存数值为 `torch.cuda.max_memory_allocated()`，不包含全部驱动、缓存或桌面占用。

这些是**训练一轮后的完整测试结果**；本次没有执行默认 200 轮训练或收敛精度复现。

## 数据与配置

| 项目 | PEMS08 | STContext NYC-Bike |
| --- | --- | --- |
| 原始数组 | `[17856,170,3]` | `[37248,820]` |
| 模型节点数 | 170 | 326，沿用首日已有记录的站点集合 |
| 输入 → 输出 | 12 → 12 个五分钟步 | 24 → 24 个小时步 |
| 训练 / 验证 / 测试窗口 | 10699 / 3567 / 3567 | 21889 / 8761 / 6505 |
| batch size | 32 | 4 |
| 历史 / 未来 / 静态协变量维度 | 2 / 4 / 3 | 7 / 5 / 10 |
| UniCA 门控 / 注意力头数 | 关闭 / 8 | 开启 / 4 |
| 每轮训练打乱 | 关闭，沿用 PEMS 启动脚本设置 | 开启 |
| 尾批 | 填充，最终测试裁掉重复样本 | 保留实际大小，不填充 |

NYC 原始数据固定 STContext commit `30615d4ffcf804cb1d8e5399ba64e2d1385b668b`，
使用源文件声明的 **60 分钟**间隔，没有再次聚合 12 个小时。
训练目标截至 2015-12-31，验证目标属于 2016 年，测试目标为
2017-01-01 至 2017-09-29；三个 split 的目标时间不相交。
历史天气、日历、2013 年 POI 与空间邻接图均按既有协议处理。

PEMS 指标只包含正目标；NYC 的 MAE/RMSE/WMAPE 包含有效的零需求，MAPE 只包含正目标。
结果 CSV 的 MAPE、WMAPE 使用比例；最后一行是各预测步指标的算术平均。

## 复现命令

从仓库根目录执行，沿用前次已缓存的 GPT-2 权重：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl
cd ~/Projects/TrafficFlowPrediction
python -m pip install -r requirements-nyc-bike.txt
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p ccfa-workfiles/checks/unica-pems08-nyc-bike

python -B download_pems.py --datasets pems08
python -B prepare_pems08.py
python -B tests/verify_prepared_data.py --data pems08 \
  --report ccfa-workfiles/checks/unica-pems08-nyc-bike/pems08-data-checks.json

python -B download_nyc_bike.py --profile training
python -B prepare_nyc_bike.py

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B -m unittest discover -s tests -v
python -B tests/smoke_training.py --data pems08 --device cuda:0 \
  --llm_layer 6 --U 1 --batch_size 32 --epochs 1 \
  --unica_enabled --unica_training_policy adapter_lora \
  --save ccfa-workfiles/checks/unica-pems08-nyc-bike/unica-
python -B tests/smoke_training.py --data nyc_bike --device cuda:0 \
  --llm_layer 6 --U 1 --input_len 24 --output_len 24 --steps_per_day 24 \
  --batch_size 4 --epochs 1 --unica_enabled --unica_with_gate --unica_heads 4 \
  --unica_training_policy adapter_lora --model_selection_split val --shuffle_train \
  --save ccfa-workfiles/checks/unica-pems08-nyc-bike/unica-
```

已有 NYC 准备结果时无需重跑准备脚本，该脚本会拒绝覆盖已有数据。
本次 NYC 逐窗口核验直接调用 `audit_nyc_bike.verify_prepared`，并检查下载清单内各输入的哈希：

```bash
python -B - <<'PY'
import json
from pathlib import Path
from audit_nyc_bike import verify_prepared
from utils.nyc_bike import sha256, write_json

raw = Path("data/nyc_bike/raw")
manifest = json.loads((raw / "download_manifest.json").read_text())
assert all(sha256(raw / r["path"]) == r["sha256"] for r in manifest["files"])
report = verify_prepared(raw, Path("data/nyc_bike_24_24"))
report["source_files_sha256_checked"] = len(manifest["files"])
write_json("ccfa-workfiles/checks/unica-pems08-nyc-bike/nyc-bike-data-checks.json", report)
PY
```

## 工程改动与检查

- NYC 下载增加 `--profile training`，下载模型所需的完整数据；默认 `full` 保留原来的额外
  上下文下载。切换 profile 会保留已有文件的来源记录。对应回归测试及其他单元测试共
  **23 项全部通过**。
- 训练检查增加 UniCA 实例与权重变化断言，记录 `adapter_changed_tensors`，确认适配器参与更新。
- 两套数据的全部窗口、协变量与邻接图核验通过；NYC 还核对站点顺序、目标时间隔离、
  流量守恒及空间特征。数据检查记录保存在本次工作目录。
- `python -m pip check` 通过。Python 3.12.16、PyTorch 2.8.0+cu128、Transformers 5.22.1、
  PEFT 0.20.0；完整环境快照为工作目录下 `environment.txt`。
- NYC 曾尝试 batch size 16，在反向传播时遇到 CUDA 内存分配失败，当时 PyTorch 已分配
  约 9.43 GiB；随后改用项目默认 batch size 4。失败日志和配置保存在
  `nyc-bike-batch16-oom.log` 与 `nyc-bike-batch16-oom/`。后续定位见下节。

工作目录为 `ccfa-workfiles/checks/unica-pems08-nyc-bike/`。
运行配置、抽样/归一化记录、训练/测试 CSV、最佳权重与检查证据保存在
`unica-pems08/` 和 `unica-nyc_bike/`；原始输出为 `pems08.log` 和 `nyc-bike.log`。
数据位于 `data/PEMS08/`、`data/pems08_12_12/`、`data/nyc_bike/raw/` 和 `data/nyc_bike_24_24/`。

## NYC batch size 16 的 OOM 定位

后续对照证据指向 **Windows 系统提交内存（commit）耗尽**，而非 GPU 的 32 GiB 容量被用满。
提交内存是系统承诺提供后备存储的内存额度，与物理 RAM 空闲量、CUDA 显存空闲量不同。
微软说明：提交上限由物理内存和页面文件共同支撑，达到上限会导致新的内存提交失败。
参见 [Windows 页面文件与系统提交内存说明](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file)。

发生上述 OOM 时，Windows 页面文件关闭自动管理，`C:\pagefile.sys` 初始 1024 MiB、最大 8192 MiB。
WSL 可见内存约 15.49 GiB。诊断保持原来的模型、FP32、batch size 16、UniCA 参数、
归一化统计和随机种子，比较完全相同的两个真实训练 batch，其输入/目标 SHA-256 一致。
三种加载条件的第一次更新损失也完全相同（3.9563072844882683）。

| 对照条件 | 结果 |
| --- | --- |
| 原始全量加载和每轮打乱 | 复现 CUDA OOM；CUDA 仍报告约 21 GiB 空闲 |
| 先全量加载，再释放除两个 batch 外的数据 | 仍 OOM；Linux 进程 RSS 已降至约 1.85 GiB，说明仅看进程 RSS 不能判断 Windows 提交余量 |
| 新进程从启动就只加载已导出的同两个真实 batch | 两次完整前向、反向和优化器更新通过；峰值 PyTorch 分配量 **9.7334 GiB** |
| 不加载模型和数据，仅分配并写入 CUDA 张量 | **12 GiB 成功**，排除约 9 GiB 的硬性 GPU 分配上限 |

失败对照中的 Windows 提交量达到 **39.7305 / 39.7475 GiB（99.957%）**，
当时只剩约 17.48 MiB 提交余量，而 CUDA 又请求分配 78 MiB。
低主存初始化的成功对照中，采样到的最大提交比例为 98.73%，未触及上限。
独立 CUDA 探测分配 12 GiB 时，Windows 提交量增加约 **12.11 GiB**；
因此 GPU 仍有空闲，并不保证 Windows/WSL 能继续为 CUDA 分配提供系统提交额度。

同时确认一个加重主存压力的代码路径：`DataLoader.shuffle()` 为训练 x/y 创建新数组，
但 `load_dataset()` 返回字典仍引用旧数组。本次实际数组占用由 **6.548 GiB 增至 10.405 GiB**，
多出约 **3.857 GiB**。这是提交压力的来源之一；先加载再删除数组的对照仍失败，
说明释放 Python 数组、降低 RSS 本身不足以保证 Windows 提交量立即下降。

处理优先级是提高 Windows 页面文件的提交支撑额度（例如改为系统管理，或提高现有 8 GiB 上限），
并减少数据窗口副本，例如按索引打乱和按 batch 取样。当次只进行诊断，未修改 Windows 页面文件、
WSL 配置或生产模型/训练实现。batch size 16 的两步诊断成功不等于已完成其完整训练。

原错误信息中的 `17179869184 GiB` 也不是真实进程占用。NVML 在 WDDM 下可以为进程显存返回
`NVML_VALUE_NOT_AVAILABLE`，而 PyTorch 2.8 的报错路径直接格式化该字段，产生了这个极大值。
参见 [NVIDIA NVML 进程显存字段](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlProcessInfo__v1__t.html)
及 [PyTorch 2.8 报错实现](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp#L1024-L1073)。

诊断代码、阶段内存记录、Windows 提交量采样、两个真实 batch 和 CUDA 内存快照均位于
`ccfa-workfiles/checks/unica-pems08-nyc-bike/oom-diagnosis/`。
主要证据为 `full.json`、`batch-only.json`、`batch-only.windows.jsonl`、
`cached-batches.json`、`cached-batches.windows.jsonl` 和 `cuda-capacity.json`。

## NYC batch size 16 复测：完整一轮通过

2026-11-03 同日复测。启动前读取到 `C:\pagefile.sys` 已设置为初始 **16384 MiB**、
最大 **32768 MiB**，实际已分配 16384 MiB；Windows 系统提交上限为 **47.7475 GiB**。
本次复测未修改页面文件、WSL 设置或生产训练代码。

使用 `conda activate dl`，继续使用完整 NYC-Bike 数据集、326 个站点、24→24 小时窗口，
FP32、GPT-2 六层、`U=1`、TFP + UniCA、门控、4 个 UniCA 注意力头、
`adapter_lora`、每轮打乱、种子 6666。完整运行的解析后配置与原 batch size 16
失败配置逐项相同，只有输出路径不同。

先执行 5 次训练更新、2 个验证 batch、2 个测试 batch 的短测并通过；随后执行
**无 batch 截断的一整轮训练、完整验证、最佳权重重载和完整测试**，进程退出码为 0。

| 检查项 | 完整运行结果 |
| --- | ---: |
| 训练 / 验证 / 测试样本数 | 21889 / 8761 / 6505 |
| batch size | 16，尾批保留实际大小 |
| 训练更新 / 验证 batch / 测试 batch | 1369 / 548 / 407 |
| 总前向次数 | 2324 |
| 更新的 UniCA 张量数 | 123 |
| 有限输出、梯度、输出形状、权重重载 | 全部通过 |
| 最后测试输出形状 | `[9,24,326,1]` |
| 峰值 PyTorch 分配显存 | **9.7336 GiB** |
| Windows 提交量采样峰值 / 当时上限 | **39.9892 / 47.7475 GiB** |
| Windows 提交余量采样最小值 | **7.7584 GiB** |
| 训练 / 验证耗时 | 437.8137 / 57.9095 秒 |
| 验证 MAE | 1.82954460 |
| 测试 MAE / RMSE | 2.02304156 / 4.13018392 |
| 测试 MAPE / WMAPE | 61.402577% / 51.701780% |

Windows 提交量约每 1.3 秒采样一次，共 435 条；该数值是采样峰值。
PyTorch 显存数值由运行内 `torch.cuda.max_memory_allocated()` 记录，不包含全部驱动和桌面占用。
此次结果证明当前环境中 batch size 16 的完整一轮可以跑通；尚未执行 200 轮训练。

复测证据统一保存在：
`ccfa-workfiles/checks/unica-pems08-nyc-bike/batch16-recheck-2026-11-03/`。
其中 `verification.json` 为自动核验结果，`full.log` 为完整输出，
`full.windows.jsonl` 为 Windows 内存采样，`windows-before.json` 为启动前配置，
`run_batch16.sh` 记录实际命令；`full-nyc_bike/` 内包含配置、训练和测试 CSV、
最佳权重与 `smoke_checks.json`。先前失败及 batch size 4 的结果继续保留在各自目录。

后续从项目根目录使用生产入口运行一轮：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash train_nyc_bike.sh unica --batch_size 16 --epochs 1 --save ./logs/nyc-unica-b16-
```

## NYC batch size 32：完整一轮通过

2026-11-03 按用户要求继续测试 `batch_size=32`。仍使用 `conda activate dl`、完整
NYC-Bike 数据、TFP + UniCA、FP32、六层 GPT-2、`U=1`、UniCA 门控和 4 个注意力头、
`adapter_lora`、每轮打乱与种子 6666。解析后的完整运行参数与 batch size 16 参考配置
逐项一致，只有 batch size 和输出路径不同。生产训练、模型和数据加载代码均未在此轮改动。

启动前读取到 WSL 的 `memory=32GB` 设置已经生效，swap 为 4 GB；
Windows 页面文件仍为初始 16384 MiB、最大 32768 MiB。本次没有修改这些系统设置。

先执行 5 次训练更新、2 个验证 batch、2 个测试 batch 的短测并通过；随后执行
**无 batch 截断的一整轮训练、完整验证、最佳权重重载和完整测试**，两次进程退出码均为 0。

| 检查项 | 完整运行结果 |
| --- | ---: |
| 训练 / 验证 / 测试样本数 | 21889 / 8761 / 6505 |
| batch size | 32，尾批保留实际大小 |
| 训练更新 / 验证 batch / 测试 batch | 685 / 274 / 204 |
| 总前向次数 | 1163 |
| 更新的 UniCA 张量数 | 123 |
| 有限输出、梯度、输出形状、权重重载 | 全部通过 |
| 最后测试输出形状 | `[9,24,326,1]` |
| 峰值 PyTorch 分配显存 | **18.7089 GiB** |
| Windows 提交量采样峰值 / 当时上限 | **53.8423 / 56.8656 GiB** |
| Windows 提交余量采样最小值 | **1.0242 GiB** |
| Windows 空闲物理内存采样最小值 | 2684 MiB |
| 训练 / 验证耗时 | 430.7779 / 61.3568 秒 |
| 验证 MAE | 1.94125934 |
| 测试 MAE / RMSE | 2.14343804 / 4.50843992 |
| 测试 MAPE / WMAPE | 62.298931% / 54.778895% |

完整运行共记录 434 条 Windows 内存采样，约每 1.3 秒一次。提交上限在运行中由
47.7475 GiB 增长到 56.8656 GiB；运行中 CIM 快照显示页面文件实际已分配 25720 MiB，
即约 25.12 GiB。短测也通过，但其页面文件增长期间采样到的最小提交余量仅约 0.260 GiB。
因此当前环境可以运行 batch size 32，系统提交额度的余量比显存余量更紧。
PyTorch 显存记录不包含全部驱动和桌面占用。这是完整一轮的可运行性验证，未执行 200 轮训练。

复测证据位于：
`ccfa-workfiles/checks/unica-pems08-nyc-bike/batch32-check-2026-11-03/`。
`verification.json` 包含自动核验、内存统计和源码 SHA-256；`full.log` 与
`full.windows.jsonl` 保存完整输出和内存采样；`windows-before.json`、
`windows-during.json`、`windows-after.json` 保存系统配置快照。
实际命令位于 `run_batch32.sh`；`full-nyc_bike/` 保存配置、训练/测试 CSV、
最佳权重及 `smoke_checks.json`。之前各 batch size 的结果均保留。

后续从项目根目录使用生产入口运行一轮：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash train_nyc_bike.sh unica --batch_size 32 --epochs 1 --save ./logs/nyc-unica-b32-
```
