# Qwen 与 LLaVA 实验运行指南

更新日期：2026-10-30。适用于服务器 `/data/X-D-Lab/ty/sentiment-report`，Conda 环境名为 `sentiment-report`。

**你现在的位置：已安装运行环境、已上传数据集（用户确认）。从第 1 步开始即可，不需要重新创建环境。** 数据在服务器上的完整性、模型下载与真实推理结果仍待执行确认。

## 先明确：这里运行的是推理评测

当前代码读取一条文本及其图片，让已经训练好的 Qwen 或 LLaVA 输出 `positive`、`negative` 或 `neutral`，然后与人工标签比较。**不会更新模型权重，没有 epoch、学习率、训练 loss 或 LoRA 训练步骤。** `run` 是运行推理，不是训练命令。

| 任务 | 命令 | 作用 |
|---|---|---|
| 检查上传数据 | `check-data` | 检查文件是否齐全、内容是否与冻结版本一致 |
| 运行 Qwen / LLaVA | `run --config ...` | 加载指定模型，逐条生成分类结果 |
| 计算准确率和 Macro-F1 | `evaluate --run ...` | 用人工标签给预测结果评分 |
| 比较两次实验 | `compare --runs ...` | 汇总指标并列出逐题变化 |
| 固定正式实验方案 | `freeze` | 保存模型、提示和数据的版本信息；正式测试前执行 |

数据没有训练集：开发集 12 条用来检查程序、选择提示；测试集 47 条用于最终结果；挑战集 11 条单独报告。如果确实要微调模型，需要另行准备训练数据和训练程序，不能直接拿这 47 条测试数据训练。

本指南按顺序执行。所有命令均在服务器 Bash 终端、项目根目录运行。代码块里的反斜杠 `\` 表示命令换行，可整块复制；不要复制终端的用户名和 `$` 提示符。

## 1. 激活环境，选择一张可以使用的 GPU

### 1.1 进入目录并激活

```bash
cd /data/X-D-Lab/ty/sentiment-report
conda activate sentiment-report
```

终端通常会出现 `(sentiment-report)` 前缀。新开终端时重新执行这两条。

### 1.2 查看 GPU 并指定一张

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
```

确认哪张卡可供自己使用。之前的截图中四张卡都有进程，不能沿用当时的占用情况判断现在是否空闲。

复制获分配的那张卡的完整 UUID（`GPU-` 开头），执行下面两条；第一条会等待你粘贴 UUID 并按回车：

```bash
read -r -p "粘贴可用 GPU 的 UUID（GPU- 开头）: " SELECTED_GPU_UUID
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_UUID"
```

这会让下面启动的模型只看到这一张卡。配置文件里的 `"device": "cuda:0"` 不用改：它代表当前可见的第一张卡。Qwen 和 LLaVA 按顺序运行，上一条命令结束、终端提示符回来后，再运行下一个模型。

## 2. 缓存是什么？只设置一次存放模型的位置

**缓存就是下载好的模型文件在服务器硬盘上的存放目录。** 它与上传的数据集、GPU 显存不同。首次运行会下载模型文件，后续使用同一缓存和同一模型版本时会复用已有文件；加载模型到显存仍然需要时间。

如果还没下载过模型，可以执行：

```bash
export HF_HOME=/data/X-D-Lab/ty/.cache/huggingface
mkdir -p "$HF_HOME"
df -h "$HF_HOME"
```

这几条命令只指定并创建目录、查看磁盘空间，**不会开始下载模型**。模型会在第 4、5 步第一次运行时自动下载。两套模型需要较大的磁盘空间，实际所需空间以下载文件为准。

如果你已经把模型下载到了原来的缓存，请继续使用原路径，不要为了照抄本段而换目录。未设置 `HF_HOME` 时，默认通常为 `~/.cache/huggingface`（也受 `XDG_CACHE_HOME` 影响）；单独设置过 `HF_HUB_CACHE` 时，它决定模型仓库缓存位置。相关定义见 [Hugging Face 官方说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables)。

新开终端后，重新设置同一个 `HF_HOME`。**Conda 清华镜像只加速环境软件包获取，不会同时改变 Hugging Face 模型下载地址。**

## 3. 安装验证是什么？确认环境和数据确实能用

“安装验证”只是检查 Python 能否使用 GPU，以及数据是否放对位置，不是重新安装，也不是训练。

### 3.1 检查 GPU（首次运行前做一次）

```bash
python - <<'PY'
import sys
import torch
import torchvision
import transformers

print("Python 路径:", sys.executable)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("PyTorch CUDA 版本:", torch.version.cuda)
print("CUDA 可用:", torch.cuda.is_available())
assert torch.cuda.is_available(), "当前环境不能使用 CUDA，请先排查环境或选卡"
print("可见 GPU 数量:", torch.cuda.device_count())
assert torch.cuda.device_count() == 1, "请先按第 1 步只选择一张 GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("支持 BF16:", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), "当前 Qwen 配置使用 BF16，请核对选中的 GPU"
print("GPU 运算检查:", torch.ones(1, device="cuda:0").item())
PY
```

预期：Python 路径属于 `sentiment-report` 环境，CUDA 可用为 `True`，GPU 数量为 `1`，名称为 A800，BF16 为 `True`，最后输出 `1.0`。这证明基础 GPU 调用可用，完整模型还需下面试跑。

### 3.2 检查数据

默认目录必须是：

```text
/data/X-D-Lab/ty/sentiment-report/datasets/mvsa_daily_v1/
├── manifest.sha256.json
├── inputs/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── challenge.jsonl
├── labels/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── challenge.jsonl
├── images/
└── ...其他冻结文件
```

上传压缩包不等于已解压到正确位置。执行：

```bash
python scripts/experiment.py check-data
```

预期输出包含 `"status": "passed"`、`"checked_files": 155`，以及 `dev: 12`、`test: 47`、`challenge: 11`。如果已经检查通过，可跳过重复检查。

如果数据放在别处，需要在后续每个 `check-data`、`run`、`evaluate`、`freeze`、`compare` 命令中追加 `--dataset /你的实际数据目录`；下面统一假设使用默认目录，避免路径混用。

## 4. 先运行 Qwen：只试 3 条

执行下面命令，会真正加载 Qwen2.5-VL-7B-Instruct，用开发集前 3 条图文做预测：

```bash
python scripts/experiment.py run \
  --config configs/qwen.locked.json \
  --split dev \
  --condition multimodal \
  --prompt baseline \
  --limit 3 \
  --out runs/qwen-smoke
```

| 参数 | 含义 |
|---|---|
| `--config configs/qwen.locked.json` | 使用固定版本的 Qwen 7B，BF16 精度 |
| `--split dev` | 使用开发集 |
| `--condition multimodal` | 同时输入图片和文本 |
| `--prompt baseline` | 使用基础提示 `prompts/baseline.txt` |
| `--limit 3` | 只跑前 3 条，用于首次检查 |
| `--out runs/qwen-smoke` | 把这次运行的记录存到这个新目录 |

首次执行通常先下载模型，再加载模型，最后才出现样本进度。示意输出为 `[1/3] 样本ID: ok`，最后汇总包含 `"state": "completed"`。这里是预期格式，不是已获得的实验结果。

`ok` 只说明模型输出能被解析为分类标签，**不代表预测正确**。如果出现 `error` 或 `invalid_output`，先查看原始回答和报错。

查看这 3 条预测：

```bash
cat runs/qwen-smoke/predictions.jsonl
```

评分并查看摘要：

```bash
python scripts/experiment.py evaluate --run runs/qwen-smoke
cat runs/qwen-smoke/evaluation/summary.md
```

摘要中 `n=3`、`scope=dev_subset` 是正常的；这个分数只用于试跑检查，不是最终模型成绩。

如果提示目录已经存在，按第 10 节选择续跑或新目录，不能直接重复覆盖。

## 5. 再运行 LLaVA：同样试 3 条

等 Qwen 命令结束后，在同一终端执行：

```bash
python scripts/experiment.py run \
  --config configs/llava.json \
  --split dev \
  --condition multimodal \
  --prompt baseline \
  --limit 3 \
  --out runs/llava-smoke
```

这次加载的是 LLaVA 1.5 7B，使用 FP16。第一次运行也可能下载它自己的模型文件。

```bash
cat runs/llava-smoke/predictions.jsonl
python scripts/experiment.py evaluate --run runs/llava-smoke
cat runs/llava-smoke/evaluation/summary.md
```

只有两套模型都能完成真实推理，再继续完整实验。不要因为 `--dry-run` 通过就认为模型运行成功：`--dry-run` 不加载模型，也不生成预测。

## 6. 跑完整开发集：每次 12 条

这一阶段检查完整运行情况，并比较 Qwen 的两套提示。删除 `--limit 3` 就会跑完整开发集。下面三条依次执行；每条结束后确认没有报错，再执行下一条。

### 6.1 Qwen：图文输入，基础提示

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --condition multimodal --prompt baseline --out runs/qwen-dev-baseline
python scripts/experiment.py evaluate --run runs/qwen-dev-baseline
cat runs/qwen-dev-baseline/evaluation/summary.md
```

### 6.2 LLaVA：图文输入，基础提示

```bash
python scripts/experiment.py run --config configs/llava.json --split dev --condition multimodal --prompt baseline --out runs/llava-dev-baseline
python scripts/experiment.py evaluate --run runs/llava-dev-baseline
cat runs/llava-dev-baseline/evaluation/summary.md
```

### 6.3 Qwen：图文输入，证据核对提示

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --condition multimodal --prompt evidence --out runs/qwen-dev-evidence
python scripts/experiment.py evaluate --run runs/qwen-dev-evidence
cat runs/qwen-dev-evidence/evaluation/summary.md
```

`evidence` 对应 `prompts/evidence.txt`，是本项目的提示改进候选，不保证分数一定更高。

### 6.4 比较开发集结果

比较两个模型的基础提示：

```bash
python scripts/experiment.py compare --runs runs/llava-dev-baseline runs/qwen-dev-baseline --out runs/dev-model-comparison
cat runs/dev-model-comparison/comparison.csv
```

比较 Qwen 的基础提示和证据核对提示：

```bash
python scripts/experiment.py compare --runs runs/qwen-dev-baseline runs/qwen-dev-evidence --out runs/qwen-dev-comparison
cat runs/qwen-dev-comparison/comparison.csv
```

检查 `n=12`、无效/缺失数量、Accuracy、Macro-F1 和具体错误样本。只在开发阶段调整提示与配置；正式测试后不能根据测试分数反复改提示再挑最好成绩。

## 7. 固定方案，再跑正式测试：每次 47 条

### 7.1 固定本次实验方案（执行一次）

这里的“固定”是把模型配置、提示词和数据哈希保存到 `plans/final.json`。它是项目代码对正式测试的要求；没有该文件，真实的 `test` / `challenge` 运行会被拒绝。

先完成第 6 步，再执行下面的输入命令。提示出现后，填写真实的开发集检查结果和选择依据，按回车；不要照抄不存在的实验数字。

```bash
read -r -p "填写开发集结果、检查结论与正式比较理由: " DEV_NOTE
python scripts/experiment.py freeze \
  --configs configs/llava.json configs/qwen.locked.json \
  --prompts baseline evidence \
  --conditions text multimodal \
  --note "$DEV_NOTE；计划比较：LLaVA与Qwen图文baseline、Qwen文字baseline与图文baseline、Qwen图文baseline与evidence；改进无效也报告。" \
  --out plans/final.json
```

预期显示 `Protocol frozen: plans/final.json`。已有该文件时不要覆盖；沿用已有一致方案，或为新实验建立 `plans/final-v2.json` 并在所有对应运行中使用新路径。

本段固定的是两套本地模型。如果课程主实验还要加入 DMX 闭源模型，应先完成其开发检查，把已验证的配置加入 `--configs` 再固定完整方案；详见[原版参考第 5、6 节](experiment-reference.md)。仅完成本指南的两套本地模型，不代表闭源对比也已完成。

### 7.2 Qwen：正式图文测试

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split test --condition multimodal --prompt baseline --protocol plans/final.json --out runs/qwen-test-mm
python scripts/experiment.py evaluate --run runs/qwen-test-mm
cat runs/qwen-test-mm/evaluation/summary.md
```

### 7.3 LLaVA：正式图文测试

```bash
python scripts/experiment.py run --config configs/llava.json --split test --condition multimodal --prompt baseline --protocol plans/final.json --out runs/llava-test-mm
python scripts/experiment.py evaluate --run runs/llava-test-mm
cat runs/llava-test-mm/evaluation/summary.md
```

### 7.4 Qwen：只输入文字，比较图片的作用

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split test --condition text --prompt baseline --protocol plans/final.json --out runs/qwen-test-text
python scripts/experiment.py evaluate --run runs/qwen-test-text
cat runs/qwen-test-text/evaluation/summary.md
```

这里仍然加载同一个 Qwen 模型，只是不传图片；人工标签和测试样本不变。

### 7.5 Qwen：使用证据核对提示

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split test --condition multimodal --prompt evidence --protocol plans/final.json --out runs/qwen-test-evidence
python scripts/experiment.py evaluate --run runs/qwen-test-evidence
cat runs/qwen-test-evidence/evaluation/summary.md
```

以上每次评分应有 `n=47`、`scope=full_split`。正式测试不允许 `--limit`。如果运行中断，评分程序仍会把缺失项计错，因此还需检查运行状态和缺失数量，不能只看样本总数。

### 7.6 生成三份比较表

```bash
python scripts/experiment.py compare --runs runs/llava-test-mm runs/qwen-test-mm --out runs/local-model-comparison
python scripts/experiment.py compare --runs runs/qwen-test-text runs/qwen-test-mm --out runs/modality-comparison
python scripts/experiment.py compare --runs runs/qwen-test-mm runs/qwen-test-evidence --out runs/prompt-comparison
```

| 输出目录 | 回答的问题 |
|---|---|
| `runs/local-model-comparison/` | LLaVA 和 Qwen 在同一测试集上表现如何？ |
| `runs/modality-comparison/` | Qwen 加入图片后，哪些题变对或变错？ |
| `runs/prompt-comparison/` | Qwen 换成证据核对提示后有什么变化？ |

每个目录的 `comparison.csv` 是指标总表，`paired_changes.csv` 是逐题变化表。命令里排在第一位的运行是比较基准。

## 8. 挑战集：单独运行 11 条

使用相同固定方案，运行以下四组。每条成功结束后继续下一条：

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split challenge --condition multimodal --prompt baseline --protocol plans/final.json --out runs/qwen-challenge-mm
python scripts/experiment.py run --config configs/llava.json --split challenge --condition multimodal --prompt baseline --protocol plans/final.json --out runs/llava-challenge-mm
python scripts/experiment.py run --config configs/qwen.locked.json --split challenge --condition text --prompt baseline --protocol plans/final.json --out runs/qwen-challenge-text
python scripts/experiment.py run --config configs/qwen.locked.json --split challenge --condition multimodal --prompt evidence --protocol plans/final.json --out runs/qwen-challenge-evidence
```

逐组评分：

```bash
python scripts/experiment.py evaluate --run runs/qwen-challenge-mm
python scripts/experiment.py evaluate --run runs/llava-challenge-mm
python scripts/experiment.py evaluate --run runs/qwen-challenge-text
python scripts/experiment.py evaluate --run runs/qwen-challenge-evidence
```

生成挑战集比较表：

```bash
python scripts/experiment.py compare --runs runs/llava-challenge-mm runs/qwen-challenge-mm --out runs/challenge-model-comparison
python scripts/experiment.py compare --runs runs/qwen-challenge-text runs/qwen-challenge-mm --out runs/challenge-modality-comparison
python scripts/experiment.py compare --runs runs/qwen-challenge-mm runs/qwen-challenge-evidence --out runs/challenge-prompt-comparison
```

每次评分应为 `n=11`。挑战集与测试 47 条分开报告，不混成一个主测试分数。

## 9. 运行结束后，结果在哪里？

以 Qwen 正式图文测试为例：

```text
runs/qwen-test-mm/
├── run.json                 # 日期、代码版本、环境、命令、数据/配置、运行状态
├── config.json              # 本次模型配置
├── protocol.json            # 正式测试使用的固定方案
├── prompt_template.txt      # 提示模板
├── requests.jsonl           # 每条输入提示与图片哈希
├── predictions.jsonl        # 原始回答、预测标签、状态、耗时、显存等
└── evaluation/              # 执行 evaluate 后才会生成
    ├── summary.md           # 优先看这份中文评分摘要
    ├── metrics.json         # 完整指标
    ├── per_sample.csv       # 每条预测与人工标签
    ├── errors.csv           # 错误样本，供人工分析
    └── confusion_matrix.csv # 混淆矩阵
```

查看摘要和运行记录：

```bash
cat runs/qwen-test-mm/evaluation/summary.md
python -m json.tool runs/qwen-test-mm/run.json
```

评分包含 Accuracy 和固定三类的 Macro-F1；无效输出、报错和缺失预测都会计错。`errors.csv` 的 `possible_reason` 与 `improvement_idea` 需要结合图片、文本和原始回答手动填写，不要把猜测写成已经验证的原因。

每次正式记录应包括日期、代码版本、环境、数据版本、配置、命令、结果、失败原因与下一步。程序自动记录运行元数据和部分错误；人工的结论及下一步应另写实验笔记。所有数据、缓存、运行结果继续保留在服务器/本地，不提交到代码仓库。

## 10. 重跑、断点续跑与常见报错

### 10.1 目录已存在：不要覆盖原结果

`--out` 要求新目录。想从头做第二次实验时，换一个目录名：

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --condition multimodal --prompt baseline --limit 3 --out runs/qwen-smoke-v2
```

随后评分和查看也要使用 `runs/qwen-smoke-v2`。

### 10.2 只是中断，想接着跑剩余样本

使用原命令、原输出目录，并追加 `--resume`。例如原 Qwen 3 条试跑中断：

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --condition multimodal --prompt baseline --limit 3 --out runs/qwen-smoke --resume
```

必须保持代码、配置、提示、数据、样本范围和正式方案一致。已有 `error` 或 `invalid_output` 记录不会被自动重试；修复问题后若要重新预测这些样本，应保留失败记录并换新目录完整重跑。JSONL 尾行损坏时，程序会拒绝续跑，需要先备份并检查。

### 10.3 评分目录也不能重复覆盖

首次评分默认创建 `evaluation/`。若此前已经评分，续跑后想重新评分，指定一个新目录：

```bash
python scripts/experiment.py evaluate --run runs/qwen-smoke --out runs/qwen-smoke/evaluation-v2
cat runs/qwen-smoke/evaluation-v2/summary.md
```

比较结果目录同样要求新路径。

### 10.4 看到这些提示时怎么办？

| 提示或现象 | 处理 |
|---|---|
| 找不到 `manifest.sha256.json` | 核对数据是否解压、是否多套了一层文件夹，或统一传入 `--dataset` |
| `Frozen file changed` / 哈希不一致 | 核对上传的冻结数据包，不修改标签来绕过校验 |
| `CUDA unavailable` / `CUDA GPU not available` | 检查 Conda 环境、GPU 选择和第 3.1 步；旧代码提示中的“4090 D”并不表示必须换显卡 |
| `CUDA out of memory` | 查看该卡当前剩余显存和其他进程；确认可用资源后再跑，不结束他人的任务 |
| 下载超时或无法连接 Hugging Face | 属于模型下载连接问题；Conda 镜像不会解决它，保留错误信息排查 |
| 下载/加载期间还没有 `[1/3]` | 样本推理尚未开始；先观察下载、加载进度，不把它当成已经训练 |
| `invalid_output` | 查看 `predictions.jsonl` 的 `raw_answer`；解析只接受单一类别或仅含 `label` 的 JSON |
| `FileExistsError` | 使用第 10.1–10.3 节的新目录或续跑方式 |
| `Configuration or prompt changed after protocol freeze` | 配置或提示与固定方案不一致，回开发阶段检查并为新实验建立新方案 |
| 终端断线 | 当前命令不保证在断线后继续；重新登录后先检查已有进程和运行记录，再决定是否续跑，避免重复启动 |

## 11. 可选操作与参考

- **只检查输入、不跑模型**：执行下面的预览命令。它不下载或加载模型，不生成预测，不能评分。常规情况下第 3 步通过后可直接进行真实 3 条试跑。
- **重新装环境、DMXAPI、固定版本、旧硬件快照和完整依据**：查阅[原版环境与实验参考](experiment-reference.md)。该文件是本次改写前的保留版本，其中“服务器安装待核验”等状态为历史记录；当前进度以[项目首页](../index.md)为准。

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --limit 3 --dry-run --out runs/preview-qwen
```

本指南的命令按项目实际命令解析、推理和评分代码核对；没有在本次文档修改中远程执行 GPU 推理，也不预先填写模型成绩。
