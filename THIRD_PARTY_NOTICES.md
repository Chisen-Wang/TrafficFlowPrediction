# 来源与许可记录

项目维护者选择暂不指定项目许可证。本次未添加 `LICENSE`，也未为既有代码更改或补授许可。

| 内容 | 本次整理的处理 |
| --- | --- |
| `models/`、`utils/`、训练和数据脚本 | 保留原 TFP 工作目录实现；原目录未提供覆盖这些文件的项目许可证 |
| `ranger21.py` | 保留原项目的 `Ranger` 优化器；实现含 RAdam、Lookahead 和 Gradient Centralization，原文件未附上游仓库、版本或许可证，不能据文件名认定为 Ranger21 官方实现 |
| GPT-2、PyTorch、Transformers、PEFT 等 | 作为外部依赖安装或下载，没有将其源码或模型权重复制进发布目录；其许可独立适用 |
| PEMS 与 STContext 数据 | 提供下载和处理脚本，不附数据副本；来源见 `docs/DATA_PROTOCOL.md` |
| `compare/` 的第三方模型 | 精简版未包含这些模型、vendor 源码或上游仓库 |

README 的 ST-LLM、TimeFilter、UniCA 名称描述研究方法与实现定位，不表示已取得对应上游项目的统一授权。
原始工作目录没有足够的逐文件来源记录，本整理包不虚构作者、论文引用、上游 commit 或许可归属。
后续确定公开复用条款时，应由维护者补齐自有代码许可证及所保留实现的来源记录。
