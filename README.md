# 图文情感分类课程项目

代码仓库：[ccathead/sentiment-report](https://github.com/ccathead/sentiment-report)（私有）。

服务器首次克隆或遇到 HTTPS TLS 报错时，见[私有仓库 SSH 克隆命令](docs/github-ssh-clone.md)，包含密钥配置、认证测试和 443 端口备用命令。

实验代码入口是 `scripts/experiment.py`。支持 LLaVA 1.5 7B、Qwen2.5-VL 7B、DMXAPI 的 Responses / Chat Completions 图像接口，以及文字单模态、图文联合、基准提示与证据核对提示。

**当前操作：[全数据基础评测指南](docs/experiment-guide.md)**。使用 `run-all`，让 Qwen 用图文输入和基础提示跑完全部 70 条并自动评分，再运行 LLaVA。不需要手动切换数据划分、执行 freeze 或额外评分；原有数据和历史结果保留。当前项目是预训练模型推理评测，没有训练或微调入口。

模型命令直接用 `CUDA_VISIBLE_DEVICES` 指定 GPU，使用 `nohup python -u` 后台运行；输出与报错写入 `runs/logs/`，并保存 PID。每个模型只加载一次，预测完自动评分；日志确认 Qwen 完成后再启动 LLaVA。整份数据的分数标记为全数据基础评测。

首次安装 Conda 环境、数据放置、DMXAPI 配置和版本依据保留在[原版环境与实验参考](docs/experiment-reference.md)。基础环境文件为 `environment.yml`；创建后仍需安装 GPU 依赖。

| 文件 | 用途 |
|---|---|
| `configs/llava.json` | 固定提交的 LLaVA 配置 |
| `configs/qwen.locked.json` | 已查询官方仓库并固定提交的 Qwen 配置 |
| `configs/dmx-*.example.json` | DMX 配置模板；实际模型 ID 需按账户验证 |
| `prompts/baseline.txt`、`prompts/evidence.txt` | 两种完整英文提示模板 |
| `src/sentiment_lab/` | 数据隔离、推理适配、实验记录和评分实现 |
| `tests/test_experiments.py` | 合成数据、模拟接口的 CPU 测试，无付费调用 |

Qwen 后台命令（将 GPU 编号 `0` 改为实际可用卡；完整操作和 LLaVA 命令见指南）：

```bash
mkdir -p runs/logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/experiment.py run-all --config configs/qwen.locked.json --out runs/qwen-all-baseline >> runs/logs/qwen-all-baseline.log 2>&1 < /dev/null &
```

`run-all` 固定图文输入与 `baseline` 提示，按原始 dev/test/challenge 文件顺序读取全部样本，并在 `evaluation/` 保存汇总评分。历史分阶段命令保留在[旧版指南](docs/staged-evaluation-guide.md)。仓库不包含图片、原始帖子、复核内容、运行输出或密钥；云端需另行放入已冻结的数据包。

## 数据构建与复核记录

本项目针对英文日常生活图文内容做positive/negative/neutral三分类。研究方向为开源与闭源多模态模型比较、文字与图文输入对照、提示策略改进。

当前使用[人工复核正式数据v1](datasets/mvsa_daily_v1/README.md)：开发12条、测试47条、挑战11条，共70条。原72条全部复核，2条排除，人工标签与导出文件逐条匹配，已通过完整性与输入标签隔离检查。[数据集v0.1](datasets/mvsa_daily_v0_1/README.md)保留为AI辅助预标注的历史版本。

入口：[项目进度](index.md)、[正式数据说明](datasets/mvsa_daily_v1/README.md)、[冻结记录](reports/2026-10-30-dataset-freeze.md)。[初版复核页](datasets/mvsa_daily_v0_1/review.html)与[原标注规范](datasets/mvsa_daily_v0_1/ANNOTATION_GUIDE.md)保留供追溯。

## 文件结构

- `datasets/`：独立版本的数据集、图片和标注。
- `configs/`：明确的选样ID与判断理由，保留排除记录。
- `scripts/`：构建、复核页生成、验证、人工标注冻结脚本。
- `templates/`：离线复核页源模板。
- `work/`：候选图文查看材料，不是正式评测输入。
- `reviews/`：用户导出的真实复核记录快照，保留原始内容与哈希。
- `reports/`：数据冻结、实验与后续分析记录。

数据构建只需Python与Pillow；离线复核页不依赖额外服务。没有为本任务调用付费模型、安装训练框架或启动GPU实验。

## 重建入口

运行位置为本项目根目录。`--source`应指向包含`data/`和`labelResultAll.txt`的已解压MVSA-Single目录。

```powershell
python .\scripts\screen_candidates.py --source ..\data_sources\MVSA\extracted_kaggle\MVSA_Single
python .\scripts\build_dataset.py --source ..\data_sources\MVSA\extracted_kaggle\MVSA_Single --out .\datasets\mvsa_daily_rebuild
python .\scripts\make_review_page.py --dataset .\datasets\mvsa_daily_rebuild
python .\scripts\validate_dataset.py --dataset .\datasets\mvsa_daily_v0_1
```

已有数据版本不会被构建或冻结脚本覆盖。重建的输入ID/划分由配置与种子决定；时间戳会变化，不能保证整个输出目录逐字节相同。

当前正式数据的核验命令：

```powershell
python .\scripts\validate_frozen_dataset.py --dataset .\datasets\mvsa_daily_v1
```

最终评测使用人工冻结版本。模型推理读取`inputs/`，评测程序另读`labels/`；不要把答案、标签建议或标注理由附加进prompt。
