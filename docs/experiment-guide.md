# 实验运行指南

日期：2026-10-30。目标：以人工冻结 v1 比较通用多模态模型的情感极性分类，分析图片贡献与提示改进。代码已实现；实际模型效果只有运行后才能报告。

## 1. 实验顺序与范围

1. 开发集 12 条用于接口检查和提示选择。首次用前 3 条检查运行，不以这 3 条代替完整开发集。
2. 主测试 47 条、挑战集 11 条分别评分；正式测试不支持 `--limit`。
3. 基准提示 `baseline`：LLaVA、Qwen、一个已验证的 DMX 闭源视觉模型都跑图文输入。
4. 模态消融：至少对 Qwen 增加 `text` 条件，与该模型同一批图文预测逐题比较，gold 保持图文联合人工标签。
5. 改进实验：先在 Qwen 的开发集检查 `evidence`，随后冻结基准与改进方案，在测试集比较。即使改进无效也报告；不要根据测试分数继续调整 prompt。第二个闭源模型、更多开源模型和微调暂不列入最小交付范围。

`evidence` 是本项目的证据核对提示变体，不是对某篇论文整套方法的完整复现。当前没有 few-shot、训练、检索或隐式调用其他模型。

## 2. 获取代码和放置数据

服务器遇到 `gnutls_handshake()` 报错时，按[私有仓库 SSH 克隆命令](github-ssh-clone.md)完成配置；该记录包含逐条输入说明及 443 端口备用方案。

```bash
git clone https://github.com/ccathead/sentiment-report.git
cd sentiment-report
```

私有仓库使用自己的 GitHub 登录或 SSH；不要把访问令牌写进 URL、笔记或代码。

将已有 `mvsa_daily_v1.zip` 单独传到自己使用的 GPU 机器并解压，使目录最终满足：

```text
sentiment-report/
  datasets/mvsa_daily_v1/
    inputs/dev.jsonl
    inputs/test.jsonl
    inputs/challenge.jsonl
    labels/dev.jsonl
    labels/test.jsonl
    labels/challenge.jsonl
    images/...
    manifest.sha256.json
    ...其他冻结文件
```

这里的数据压缩包属于先前已生成的本地材料，不在 GitHub 仓库中。数据源许可和用户复核内容的公开发布尚未确认，所以代码仓库只提供工具与版本信息。`configs/curation_v0_1.json` 和 `reviews/` 同样仅保留本地；如果从最初素材重建，还需本地的选样配置与复核导出。正常复现实验直接用冻结 v1 包。

正式数据 manifest SHA-256：`fb1ef169f7237f5c6ed6039398e3b302390144265231952ce4d402da53cdb8f8`。`check-data` 应核验 155 个文件、dev=12、test=47、challenge=11。

## 3. Conda 环境与 A800 服务器

2026-10-30 更新：用户提供的服务器为 4 张 NVIDIA A800 80GB PCIe，驱动 `550.127.05`，`nvidia-smi` 显示 CUDA `12.4`。此前指南面向 4090 D；现将默认部署步骤改为 A800 + Conda，保留原有 PyTorch/Transformers 版本方案。服务器安装和真实模型推理仍待核验。

### 3.1 创建环境（首次执行）

在服务器终端进入已经克隆的项目目录。Conda 管理独立 Python 环境，GPU 软件包在激活后的环境中通过 `python -m pip` 安装。

```bash
cd /data/X-D-Lab/ty/sentiment-report
conda env create -f environment.yml
conda activate sentiment-report
python -V
python -m pip --version
```

`environment.yml` 只创建 Python 3.10 和 pip 基础环境，GPU 依赖继续按下节安装。2026-10-30 按用户要求，将项目 channel 从上游 conda-forge 改为清华 TUNA 的 conda-forge 镜像，并用 `nodefaults` 避免额外访问默认源。服务器此前停在索引下载阶段并被 Ctrl+C 中断；网络原因、镜像可达性与提速效果待核验。此修改不更改用户全局 `.condarc`。

如果服务器还没有同步此文件，可以用下面的命令代替 `conda env create`（两种创建方式只选一种）；`--override-channels` 确保本次创建只使用指定镜像：

```bash
conda create -n sentiment-report python=3.10 pip --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge -y
conda activate sentiment-report
```

已有同名环境时先用 `conda env list` 检查，确认是本项目环境后直接激活。若提示需先初始化 Conda，当前 Bash 会话可执行 `source /home/c1-505/anaconda3/etc/profile.d/conda.sh` 后再激活；路径依据用户提供的进程路径推定，实际安装位置待核验。

### 3.2 安装依赖

本地模型推理执行：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -m pip check
```

该 PyTorch/torchvision CUDA 12.4 组合有官方 wheel；驱动版本满足此 CUDA 运行时的要求。`nvidia-smi` 的 CUDA 字段表示驱动支持的 CUDA 版本，不证明系统已安装同版本 Toolkit；本项目使用预编译包推理，无需另行安装系统 CUDA Toolkit。完整依赖安装是否成功仍以服务器 `pip check` 和下面的实际 GPU 检查为准。

如果只调用 DMXAPI、核验数据或评分，可只安装轻量依赖：

```bash
python -m pip install -r requirements.txt
```

### 3.3 选择一张获分配的 GPU

用户提供的 **2026-10-30 15:46:59** 快照（后续占用可能变化）：

| GPU 编号 | 已用 / 总显存（MiB） | GPU 利用率 | 判断 |
|---|---|---|---|
| 0 | 2878 / 81920 | 99% | 显存占用低，但计算繁忙 |
| 1 | 61235 / 81920 | 0% | 已有 Python 进程大量占用显存，不代表空闲 |
| 2 | 2972 / 81920 | 99% | 显存占用低，但计算繁忙 |
| 3 | 9439 / 81920 | 100% | 计算繁忙 |

这份快照没有明确空闲卡。先确认可用卡或等待任务释放；不要根据一次 `GPU-Util=0%` 判断卡可独占。运行前查看最新状态：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
```

从输出复制**实际分配给自己**的 GPU UUID，输入以下命令后按提示粘贴并回车。这种方式避免将示例卡号误当作可用卡号：

```bash
read -r -p "粘贴获分配的 GPU UUID（GPU- 开头）: " SELECTED_GPU_UUID
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_UUID"
```

`CUDA_VISIBLE_DEVICES` 限制当前进程可见的卡，不会预留 GPU 或阻止其他用户使用。仅暴露一张卡后，它在 PyTorch 中编号为 `cuda:0`，所以 `configs/qwen.locked.json` 和 `configs/llava.json` 的 `device: cuda:0` 可以保留；不需要改成物理卡号，也不需要启用四卡并行。

### 3.4 缓存与安装验证

```bash
export HF_HOME=/data/X-D-Lab/ty/.cache/huggingface
mkdir -p "$HF_HOME"
python -c "import sys,torch,torchvision,transformers; print('Python:',sys.executable); print('torch:',torch.__version__,'torchvision:',torchvision.__version__,'transformers:',transformers.__version__); print('CUDA runtime:',torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count()==1, 'Select exactly one GPU'; print('GPU:',torch.cuda.get_device_name(0)); print('BF16:',torch.cuda.is_bf16_supported()); print('free,total GiB:',[round(v/1024**3,2) for v in torch.cuda.mem_get_info()]); x=torch.ones(1,device='cuda:0'); print('CUDA tensor:',x.item())"
python -m unittest discover -s tests -v
python scripts/experiment.py check-data
```

预期 Python 路径指向 `sentiment-report` Conda 环境，CUDA runtime 为 `12.4`，GPU 为 A800，BF16 为 `True`；若断言失败先排查，不继续启动模型。`check-data` 需要已传入冻结数据包。上述检查只验证基础环境，模型是否成功加载仍以第 4 节的 3 条开发样本运行结果为准。

新开终端需重新 `conda activate sentiment-report`，并设置 `CUDA_VISIBLE_DEVICES` 和 `HF_HOME`。可将本次环境证据保存到一个新的目录（同一终端内执行）：

```bash
ENV_RECORD_DIR="runs/env-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ENV_RECORD_DIR"
git rev-parse HEAD > "$ENV_RECORD_DIR/commit.txt"
git status --short > "$ENV_RECORD_DIR/worktree-status.txt"
conda env export > "$ENV_RECORD_DIR/conda-environment.yml"
python -m pip freeze > "$ENV_RECORD_DIR/pip-freeze.txt"
nvidia-smi > "$ENV_RECORD_DIR/nvidia-smi.txt"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$ENV_RECORD_DIR/visible-gpu.txt"
```

有独立大磁盘时可先设置 `HF_HOME` 指向缓存目录；模型会从 Hugging Face 下载或复用该目录缓存。代码采用单模型、单张图片、batch=1、贪心生成，不同时加载两套模型。LLaVA 使用 FP16，Qwen 使用 BF16；报告按各自配置记录精度。LLaVA 使用自己的图像预处理；Qwen 限制为 256–1280 视觉 token 对应像素范围。不同模型的视觉处理不同，报告需注明，不能宣称算力或分辨率完全相同。

已固定模型提交：

| 模型 | revision |
|---|---|
| `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

以后要更新版本，创建新配置并重新做开发检查，不能覆盖已有实验配置。`lock-model` 可查询公开提交而不下载权重：

```bash
python scripts/experiment.py lock-model --config configs/qwen.json --out configs/qwen-new.locked.json
```

## 4. 先预览，再运行开发集

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --limit 3 --dry-run --out runs/preview-qwen
python scripts/experiment.py run --config configs/llava.json --split dev --limit 3 --out runs/llava-smoke
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --limit 3 --out runs/qwen-smoke
```

预览不加载模型、不读取 gold、不调用 API。真实 smoke run 需检查 `predictions.jsonl` 的完整回答、状态、视觉张量形状、耗时和显存，再运行完整开发集：

```bash
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --prompt baseline --out runs/qwen-dev-baseline
python scripts/experiment.py run --config configs/qwen.locked.json --split dev --prompt evidence --out runs/qwen-dev-evidence
python scripts/experiment.py evaluate --run runs/qwen-dev-baseline
python scripts/experiment.py evaluate --run runs/qwen-dev-evidence
python scripts/experiment.py compare --runs runs/qwen-dev-baseline runs/qwen-dev-evidence --out runs/qwen-dev-comparison
```

每个 `--out` 都必须是新目录，避免覆盖历史结果。目录已存在时，断点续跑用原命令加 `--resume`；代码、配置、prompt、数据、样本范围和协议必须一致。已记录的错误/无效回答也不会自动重试；修改配置后另起完整运行并保留原失败记录。进程如果在单条 JSONL 写入中途被强制杀死，程序拒绝自动截断文件，先备份再处理尾行。

## 5. DMXAPI 配置与模型核验

1. 本地复制 `.env.example` 为 `.env`，在编辑器填写 `DMX_API_KEY`。程序仅在指定 `--env-file .env` 时读取它，也可使用同名环境变量。不需要把密钥发送到聊天。
2. 查询当前令牌可见模型。这是模型列表请求，没有推理：

```bash
python scripts/experiment.py models --env-file .env --out runs/dmx-models.json
```

3. 复制 `configs/dmx-responses.example.json` 为 `configs/dmx.local.json`，将 `model` 换成账户列表中已经核对的闭源视觉模型 ID。若账户支持的是 Chat Completions，改用 `dmx-chat.example.json`。不要仅根据模型名称认定路由支持图片。
4. 按平台与所选型号的实际支持情况设置参数：`temperature: null` 表示不发送该参数；需要关闭 GPT 推理且型号支持时可设 `extra_body` 为 `{"reasoning":{"effort":"none"}}`（Responses）或 `{"reasoning_effort":"none"}`（Chat）。不支持的参数会导致开发检查报错，需改配置，不自动静默删除。Chat 路由若只接受 `max_tokens`，把 `chat_token_parameter` 改为 `max_tokens`。
5. 确认平台余额和所选型号价格后，运行少量开发样本；下面的真实 `run` 会产生 API 调用与费用：

```bash
python scripts/experiment.py run --config configs/dmx.local.json --env-file .env --split dev --limit 3 --dry-run --out runs/dmx-preview
python scripts/experiment.py run --config configs/dmx.local.json --env-file .env --split dev --limit 3 --out runs/dmx-smoke
python scripts/experiment.py evaluate --run runs/dmx-smoke
```

代码确实把本地图片编码为 image 内容传输，不是只发送文件路径。应检查成功返回的模型名、输出、usage 与路由，再做完整开发集。HTTP 400/401/403/404 等配置错误会记录并停止；429、5xx、网络错误最多重试两次。超时重试可能重复计费，所有尝试都会记录。API 预算、账户型号及视觉能力当前仍待实际核验。

`max_output_tokens=512` 是接口输出预算，推理型模型可能把预算消耗在不可见推理上。若返回 incomplete/length，计为无效，先在开发阶段调整预算。不同模型默认采样行为不一定相同；报告按保存的实际配置说明。usage 留原始结构，不凭不确定的平台单价编造金额。

## 6. 冻结实验方案，再跑正式数据

开发阶段完成后，写清选择理由。下例只冻结两套本地模型，API 型号确认后可以把 `configs/dmx.local.json` 加入同一命令。冻结会保存配置及提示哈希，之后变动必须建立新协议并解释；不把它当作密码学意义的防篡改预注册。

```bash
python scripts/experiment.py freeze --configs configs/llava.json configs/qwen.locked.json --prompts baseline evidence --note "填写开发集检查结论、主实验组合和Qwen改进比较理由" --out plans/final.json
python scripts/experiment.py run --config configs/llava.json --split test --prompt baseline --protocol plans/final.json --out runs/llava-test-mm
python scripts/experiment.py run --config configs/qwen.locked.json --split test --prompt baseline --protocol plans/final.json --out runs/qwen-test-mm
python scripts/experiment.py run --config configs/qwen.locked.json --split test --condition text --prompt baseline --protocol plans/final.json --out runs/qwen-test-text
python scripts/experiment.py run --config configs/qwen.locked.json --split test --prompt evidence --protocol plans/final.json --out runs/qwen-test-evidence
```

闭源配置被纳入协议后：

```bash
python scripts/experiment.py run --config configs/dmx.local.json --env-file .env --split test --prompt baseline --protocol plans/final.json --out runs/dmx-test-mm
```

挑战集把 `--split test` 改成 `--split challenge`，输出目录也改为新名称；单独报告，不与测试 47 条混合得出主 Accuracy。协议是允许的配置/提示范围，`--note` 应明确实际要跑的组合，没计划的组合不因“允许”而必须跑完。

## 7. 评分与报告材料

```bash
python scripts/experiment.py evaluate --run runs/qwen-test-mm
python scripts/experiment.py evaluate --run runs/qwen-test-text
python scripts/experiment.py evaluate --run runs/qwen-test-evidence
python scripts/experiment.py compare --runs runs/qwen-test-text runs/qwen-test-mm --out runs/modality-comparison
python scripts/experiment.py compare --runs runs/qwen-test-mm runs/qwen-test-evidence --out runs/prompt-comparison
python scripts/experiment.py compare --runs runs/llava-test-mm runs/qwen-test-mm runs/dmx-test-mm --out runs/model-comparison
```

最后一条只在三组真实运行均完成后使用；不要拿 dry-run 或模拟测试当模型结果。

输出内容：

- 每次运行：`run.json`（代码 commit/源码哈希、环境、命令、配置、样本 ID、数据与提示哈希、运行状态），`prompt_template.txt`，`requests.jsonl`（完整实际用户提示及图片哈希），`predictions.jsonl`（原始回答、解析标签、耗时、API 原始响应或本地显存信息）。本地模型另记录聊天模板及最终包装后的 prompt。
- 每次评分：`metrics.json`、`summary.md`、`per_sample.csv`、`errors.csv`、`confusion_matrix.csv`。
- 对照：`comparison.csv`、`comparison.json`、`paired_changes.csv`。逐题区分 corrected/new_error/both_correct/both_wrong，便于解释图片和改进提示到底改正了什么、又引入了什么错误。

评分规则：Accuracy 的分母是完整所选 gold 集；无效输出、API 错误、缺失预测都计错。Macro-F1 固定对 positive/negative/neutral 三类求平均；无效输出计对应 gold 类的 FN。重复 ID、未知 ID 和数据版本不符直接拒绝评分。严格解析只接受单一标签（忽略大小写和外围空白）或仅含 `label` 的 JSON；不会从一段解释里搜索有利的类别词。

`errors.csv` 的原因与改进列留空，结合原图和文本填写至少两个真实案例，不能自动猜测模型内部原因。错误原因是分析假设，需要与可见证据区分。小样本、单人复核、AI 辅助预标注、样本筛选偏差与可能的模型预训练数据重合应在报告中说明。

## 8. 核验依据

- [清华 TUNA Anaconda 镜像说明](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

- [Conda 环境管理](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [PyTorch 历史版本安装命令（2.6.0 / CUDA 12.4）](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

- [LLaVA 官方模型卡](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Qwen2.5-VL 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [DMX Chat Completions 本地图片](https://doc.dmxapi.cn/base64-image.html)
- [DMX Responses 本地图片](https://doc.dmxapi.cn/res-base64-image.html)
- [OpenAI Docs 图像输入](https://developers.openai.com/api/docs/guides/images-vision)

适配采用官方公开格式与既有 LLaVA 成功记录。CPU 测试验证程序逻辑，不能代替两套真实 GPU 模型和账户 API 的端到端运行验证。
