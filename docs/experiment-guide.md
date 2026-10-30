# 全数据基础评测：先 Qwen，再 LLaVA

更新日期：2026-10-30。当前环境和数据已准备好，Qwen 的部分验证已完成（用户确认）。现在只做一件事：**每个模型用相同的基础提示，读取图片和文本，把全部 70 条数据完整预测一次并自动评分。**

## 1. 准备

在服务器执行：

```bash
cd /data/X-D-Lab/ty/sentiment-report
git pull origin main
conda activate sentiment-report
mkdir -p runs/logs
```

本页命令要求更新后的 `run-all` 功能。数据默认位于 `datasets/mvsa_daily_v1/`，不需要移动或重新划分数据。此前的 `qwen-dev-baseline`、`qwen-dev-evidence` 等结果保留；本次使用新的输出目录。

下面命令中的 `CUDA_VISIBLE_DEVICES=0` 是示例，**把 0 改成你确认可用的 GPU 编号**，也可直接填写 GPU UUID。模型配置中的 `cuda:0` 不用改。保持已有模型缓存路径即可，不需要清理或重新设置缓存。

## 2. Qwen：一条命令跑完全部数据并评分

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/experiment.py run-all \
  --config configs/qwen.locked.json \
  --out runs/qwen-all-baseline \
  >> runs/logs/qwen-all-baseline.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-all-baseline.pid
```

这条命令会自动完成：

1. 检查全部输入、图片和人工标签的对应关系及哈希。
2. 加载 Qwen2.5-VL-7B-Instruct（模型只加载一次；缺少权重时先下载）。
3. 按原文件顺序读取全部 70 条图片与文本，固定使用 `prompts/baseline.txt`。
4. 保存每条预测，随后自动计算 Accuracy、Macro-F1、混淆矩阵和错误样本。

后台启动后终端立即返回，运行输出和报错全部写入日志。先只启动 Qwen；完成后再启动 LLaVA。

查看最近 50 行进度：

```bash
tail -n 50 runs/logs/qwen-all-baseline.log
```

需要实时查看时用：

```bash
tail -f runs/logs/qwen-all-baseline.log
```

按 Ctrl+C 只退出日志查看，不会停止后台模型。日志中会逐条出现 `[1/70]` 到 `[70/70]`；最后输出包含 `"state": "completed"`、`"samples": 70` 及指标，表示推理和自动评分已完成。这里是预期格式，不是已经取得的模型成绩。

完成后查看 Qwen 评分摘要：

```bash
cat runs/qwen-all-baseline/evaluation/summary.md
```

应看到 `n=70`。同时检查无效/缺失数量；`ok` 只表示输出能解析为标签，是否预测正确由评分判断。

## 3. Qwen 完成后，再运行 LLaVA

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/experiment.py run-all \
  --config configs/llava.json \
  --out runs/llava-all-baseline \
  >> runs/logs/llava-all-baseline.log 2>&1 < /dev/null &
echo $! > runs/logs/llava-all-baseline.pid
```

这条命令用 LLaVA 1.5 7B 跑相同的 70 条图文、相同的基础提示，结束后自动评分。与 Qwen 一样，不需要再单独执行 `evaluate`。

查看进度：

```bash
tail -n 50 runs/logs/llava-all-baseline.log
```

完成后查看结果：

```bash
cat runs/llava-all-baseline/evaluation/summary.md
```

## 4. 结果保存在哪里？

| 内容 | Qwen | LLaVA |
|---|---|---|
| 完整运行日志 | `runs/logs/qwen-all-baseline.log` | `runs/logs/llava-all-baseline.log` |
| 每条原始回答、预测、耗时等 | `runs/qwen-all-baseline/predictions.jsonl` | `runs/llava-all-baseline/predictions.jsonl` |
| 日期、代码、环境、配置、命令、运行状态 | `runs/qwen-all-baseline/run.json` | `runs/llava-all-baseline/run.json` |
| 评分摘要 | `runs/qwen-all-baseline/evaluation/summary.md` | `runs/llava-all-baseline/evaluation/summary.md` |
| 完整指标 | `runs/qwen-all-baseline/evaluation/metrics.json` | `runs/llava-all-baseline/evaluation/metrics.json` |
| 逐条预测与人工标签 | `runs/qwen-all-baseline/evaluation/per_sample.csv` | `runs/llava-all-baseline/evaluation/per_sample.csv` |
| 错误样本 | `runs/qwen-all-baseline/evaluation/errors.csv` | `runs/llava-all-baseline/evaluation/errors.csv` |

同一评分目录还包含 `confusion_matrix.csv`。模型输入只有图片和文本，人工标签单独用于校验和评分，不放进提示。无效回答、错误和缺失项计错，不从分母排除。

## 5. 中断或目录已存在时

先确认原任务已经退出，避免同一目录重复运行。查看 Qwen 进程：

```bash
ps -p "$(cat runs/logs/qwen-all-baseline.pid)" -o pid,etime,stat,args
```

进程消失不代表成功，需要同时查看日志。PID 可能被复用，先核对实际命令。

**仅中断、代码和配置未变**：用原命令追加 `--resume`，接着预测还没有记录的样本，完成后自动评分：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/experiment.py run-all \
  --config configs/qwen.locked.json \
  --out runs/qwen-all-baseline \
  --resume \
  >> runs/logs/qwen-all-baseline.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-all-baseline.pid
```

LLaVA 续跑同理：替换配置、结果目录、日志与 PID 名称。

**已经完整完成，或修复报错后想全部重跑**：使用新目录，例如 `runs/qwen-all-baseline-v2`，并更换对应日志/PID 名称；不要带 `--resume`。已有错误/无效预测不会被续跑自动重试，原结果保留。

如果数据不在默认位置，在启动命令的日志重定向 `>>` 之前加 `--dataset /实际数据目录`。该目录应直接包含 `inputs/`、`labels/`、`images/` 和 `manifest.sha256.json`。

## 关于之前的“开发集”和“正式测试”

开发集原本用于调提示和检查运行，正式测试原本指保留另一批样本，在方法确定后测效果。你当前要做的是先获取模型在整份数据上的基础表现，因此本页统一评测全部 70 条，不要求你分阶段操作。

数据原来的 12/47/11 文件组织保留，`run-all` 会在内存中按顺序合并读取；不会修改原始数据或标签。这里得到的分数写作“全数据基础评测”，包含此前用过的开发样本，不称为独立保留测试集成绩。后续如果需要比较更多方法，再另行制定实验方案。

当前仍是使用预训练模型进行分类推理，不更新权重。此前多条件操作指南已保留在[旧版分阶段评测指南](staged-evaluation-guide.md)，安装和 API 资料见[环境与实验参考](experiment-reference.md)，无需为了本次基础评测执行其中的额外步骤。
