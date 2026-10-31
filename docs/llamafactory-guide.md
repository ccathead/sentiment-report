# LLaMA-Factory 微调准备与运行

> 历史方案，2026-10-31用户已改选直接PEFT。当前请使用[PEFT训练与评测指南](peft-guide.md)、scripts/train_lora.py和configs/training/，不执行本页安装或训练命令。以下保留原方案用于追溯。

日期：2026-10-31。用户已确认另建训练/验证集，保留现有70条为固定评测集，使用LLaMA-Factory。以下为准备好的配置和命令，尚未在A800上实跑，不能视为已完成训练。

当前仅三种划分：训练集train、验证集val、评测集eval。评测数据位于`datasets/mvsa_eval_v1/`，已从原70条无损汇总；不再在当前流程中区分开发与挑战样本。训练/验证候选复核页已改为动态计数与中文用途说明，不提供不存在的AI联合标签。

## 1 正式数据已生成，无需新增复核

2026-10-31用户明确决定不复核新增候选，已按来源文字／图片标签一致性生成正式数据：`datasets/mvsa_sft_v1/`，train300条（每类100）、val60条（每类20）。标签均从原MVSA-Single文件核对取得，明确标为来源一致性派生、未经人工联合复核；现有70条评测标签不变。当前直接使用这份正式数据，后文候选人工复核流程仅作为备选记录。

LLaMA-Factory两个YAML已指向该目录。将整个`datasets/mvsa_sft_v1/`目录复制到服务器`/data/X-D-Lab/ty/sentiment-report/datasets/mvsa_sft_v1/`后，直接从第2节准备训练环境，再执行两步检查和完整训练。图片子目录也必须复制；目前未连接服务器，未替用户上传或启动训练。

再生成同样数据的命令（已有输出时勿重复执行）：

```bash
python scripts/prepare_finetuning.py export \
  --packet datasets/mvsa_finetuning_candidates_v1 \
  --label-policy source-agreement \
  --source-labels ../data_sources/MVSA/extracted_kaggle/MVSA_Single/labelResultAll.txt \
  --out datasets/mvsa_sft_v1
```

来源一致性标签可能有噪声，也不代表已核验题材和近重复；与人工联合评测标签的口径差异要记录在报告中。正式文件和标签来源详见数据目录README与split_manifest.json。

### 备选：人工复核流程（本次不执行）

本地已生成 `datasets/mvsa_finetuning_candidates_v1/review.html`，直接在浏览器打开。360条候选按来源文字/图片一致标签各抽120条，**不是360条人工gold**，也不保证最终类别平衡或全部符合本项目日常题材。

填写复核人，逐条查看图片和配文，选择 positive / negative / neutral；不合题材、广告、无法理解、混合情感难裁决、近重复的条目选排除并写原因。中性表示作者主要客观陈述，不能仅因难判断就选中性。当前没有填写AI联合标签，也没有把来源标签自动用作训练答案。

准备程序已按来源ID、图片SHA256、HTML解码/NFKC/大小写与空白归一后的配文排除冻结70条、历史排除的2条以及候选内部的精确重复；**裁剪图、换句复述等近重复尚待人工检查**。来源文本无法严格UTF-8解码的样本被跳过，ID保存在metadata.json，未静默替换乱码。这样得到的候选存在来源一致性和编码筛选偏差，报告需保留这一局限。

全部复核后导出JSON，放入 `reviews/finetuning_v1.json`。从项目根目录执行（Windows和Linux同样适用）：

```bash
python scripts/prepare_finetuning.py export \
  --packet datasets/mvsa_finetuning_candidates_v1 \
  --reviews reviews/finetuning_v1.json \
  --out datasets/mvsa_sft_v1
```

程序要求每条候选都有一次完整复核，排除项也需记录；拒绝版本不匹配、图片被改动、与评测重叠、重复或缺失复核、覆盖已有输出。按人工最终标签分层，约5/6训练、1/6验证，种子20261031；如果360条全部保留，大致为300/60，实际以split_manifest.json为准。每类至少需2条才可划分，这只是程序要求，不表示样本量足够。

导出包括train.json、val.json、dataset_info.json、图片和带标签来源/哈希的split_manifest.json。采用Alpaca多模态格式，一张实际图片对应一个`<image>`占位符，assistant只监督最终标签。所有数据、复核与日志留本地。上传到服务器时保留目录结构；在 `sentiment-report` 根目录执行后续命令。

## 2 单独创建训练环境

本次固定可核验的 **LLaMA-Factory v0.9.3**，不是宣称它是最新版本。该版本要求Transformers不高于4.52.4、Accelerate不高于1.7.0，与当前评测环境4.57.1/1.15.0不一致，所以另建环境。依据：[该版本依赖](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/requirements.txt)。

Linux服务器执行一次，以下安装需要联网；模型缓存可以继续共用：

```bash
conda create -n sentiment-train python=3.10 -y
conda activate sentiment-train
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
git clone --depth 1 --branch v0.9.3 https://github.com/hiyouga/LLaMA-Factory.git /data/X-D-Lab/ty/LLaMA-Factory-v0.9.3
python -m pip install -e /data/X-D-Lab/ty/LLaMA-Factory-v0.9.3
python -m pip check
cd /data/X-D-Lab/ty/sentiment-report
mkdir -p runs/logs runs/finetuning
python -m pip freeze > runs/logs/training-environment.txt
git -C /data/X-D-Lab/ty/LLaMA-Factory-v0.9.3 rev-parse HEAD > runs/logs/llamafactory-commit.txt
git rev-parse HEAD > runs/logs/project-training-commit.txt
git status --short > runs/logs/project-training-status.txt
```

安装速度、镜像连通性和CUDA训练兼容性尚未实测。若安装失败保留日志再处理，不在原`sentiment-report`环境中直接升级或降级依赖。

## 3 先做两步训练检查

只有新数据已导出后再执行。先检查GPU2的空闲资源：

```bash
nvidia-smi -i 2 --query-gpu=name,memory.used,memory.free --format=csv
```

Qwen小规模检查命令，输出均写日志：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup llamafactory-cli train configs/finetuning/qwen_lora.yaml \
  max_samples=2 max_steps=2 do_eval=false load_best_model_at_end=false \
  output_dir=runs/finetuning/qwen-smoke-v1 \
  > runs/logs/qwen-lora-smoke-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-lora-smoke-v1.pid
```

LLaVA用同一命令结构，将配置换成`configs/finetuning/llava_lora.yaml`，输出及日志中的`qwen`换成`llava`。这项检查会实际做两步训练，但不用于最终评分。重复检查时另取新输出名，避免覆盖。

进入完整训练前，核对日志中的实际模型revision、可训练LoRA模块仅在语言分支、loss有限且非空、图片加载成功；进一步检查tokenized样本：用户/图片/padding被mask，答案标签有监督，完整图片token和答案没有被cutoff_len=4096截断。4096只是待验证的长度上限。两步命令成功不等于这些内容已全部核验。

## 4 先训练Qwen，完成后训练LLaVA

配置均采用LoRA r16/alpha32/dropout0.05、语言注意力投影、冻结视觉塔和投影器、batch1累计8、学习率5e-5、3轮。Qwen模板`qwen2_vl`，LLaVA模板`llava`；固定现有模型revision。依据：[官方Qwen训练示例](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/examples/train_lora/qwen2_5vl_lora_sft.yaml)、[模型注册](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/extras/constants.py)、[多模态冻结实现](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/model/model_utils/visual.py)。

Qwen完整训练：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup llamafactory-cli train configs/finetuning/qwen_lora.yaml \
  > runs/logs/qwen-lora-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-lora-v1.pid
```

确认Qwen结束、输出完整、释放显存后，再运行LLaVA：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup llamafactory-cli train configs/finetuning/llava_lora.yaml \
  > runs/logs/llava-lora-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/llava-lora-v1.pid
```

查看日志末尾即可，不需要把训练输出留在终端：

```bash
tail -n 40 runs/logs/qwen-lora-v1.log
ps -p "$(cat runs/logs/qwen-lora-v1.pid)" -o pid,etime,args
```

当前YAML自动按**独立验证集eval_loss最低**选择checkpoint，未实现生成式分类Macro-F1选优，也不能把token accuracy当作情感分类准确率。若后续改为按生成Macro-F1挑选，需另加验证推理并记录选择规则；禁止根据70条回归成绩挑checkpoint。

当前项目的`experiment.py`还不能直接读取LoRA适配器。因此训练结束不等于完成70条评测，**不要把旧qwen.locked.json命令直接重跑当成微调结果**。后续需实现并验证适配器加载或合并模型入口，核对训练/评测框架版本间权重映射、聊天模板和图像处理差异，再固定同一baseline任务提示，在70条上各完整评测一次。此阶段没有编造可用的适配器评测命令。

## 5 API模型共用新数据，但不微调

DeepSeek和Kimi的few-shot示例仅从新训练集选，每类2条并发送真实图片；新验证集比较baseline、boundary、boundary+6-shot，固定后才跑原70条。当前单图API消息还需扩展，不会把`boundary_v1.txt`直接称为已完成多图few-shot。

DeepSeek原基线mvsa_2533的512个输出token全用于reasoning，最终答案空且finish_reason=length；工程修复可先把预算作为单独变量增加并在新验证集检查，实际阈值待核验。修复后的baseline和提示优化应使用一致预算，单独保留旧结果与修复记录。
