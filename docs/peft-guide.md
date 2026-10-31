# 直接使用 Transformers + PEFT 微调

更新日期：2026-10-31。当前采用PEFT LoRA，不需要LLaMA-Factory、TRL、WebUI或合并7B权重。历史LLaMA-Factory配置保留作记录，本指南是当前入口。

已完成代码和本地CPU检查，**尚未在服务器A800上训练**。300条训练、60条验证使用MVSA原文字/图片一致性标签，未经人工联合复核；70条评测沿用人工复核标签。不得把数据准备完成写成已取得微调提升。

## 1 放好数据和环境

在服务器项目根目录执行。`datasets/mvsa_sft_v1/`需从本地完整复制到服务器，包含train.json、val.json、dataset_info.json、split_manifest.json和images/；Git不会上传这些文件。评测数据使用`datasets/mvsa_eval_v1/`，若尚未生成，先运行`python scripts/consolidate_evaluation.py`，保留原70条文件供脚本汇总。

推荐克隆已跑通的Conda环境，只添加PEFT，保留原评测环境：

```bash
cd /data/X-D-Lab/ty/sentiment-report
conda create -n sentiment-peft --clone sentiment-report -y
conda activate sentiment-peft
python -m pip install -r requirements-training.txt
python -m pip check
mkdir -p runs/logs
python -m pip freeze > runs/logs/peft-environment.txt
git rev-parse HEAD > runs/logs/peft-code-commit.txt
git status --short > runs/logs/peft-code-status.txt
```

沿用已验证的PyTorch 2.6.0+cu124、Transformers 4.57.1、Accelerate 1.15.0，新增固定PEFT 0.17.1；安装需要联网，之后模型从原缓存离线加载。已有同名环境时直接激活，不重复create。若当前环境不是原GPU评测环境，应先核对torch CUDA版本，requirements-training.txt不会替你安装CUDA版torch。

先检查文件，不加载权重、不占GPU：

```bash
python scripts/train_lora.py --config configs/training/qwen_lora.json --check-only
python scripts/train_lora.py --config configs/training/llava_lora.json --check-only
nvidia-smi -i 2 --query-gpu=name,memory.used,memory.free --format=csv
```

两份检查应显示train300/val60、label_policy=source-agreement。它们校验文件哈希、输入与标签来源清单、训练/验证/评测之间的相同ID、图片及归一配文重复；不使用评测标签来训练或挑checkpoint。

## 2 每个模型先做两步检查

Qwen：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup python -u scripts/train_lora.py \
  --config configs/training/qwen_lora.json \
  --out runs/qwen-lora-smoke-v1 --smoke \
  >> runs/logs/qwen-lora-smoke-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-lora-smoke-v1.pid
```

LLaVA：等前一个进程结束后再执行。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup python -u scripts/train_lora.py \
  --config configs/training/llava_lora.json \
  --out runs/llava-lora-smoke-v1 --smoke \
  >> runs/logs/llava-lora-smoke-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/llava-lora-smoke-v1.pid
```

这两次仅做2个优化步骤，用于检查真实模型的图片输入、答案mask、反向梯度和显存。查看日志有`finite nonzero LoRA gradients`且run.json为completed再进行完整训练。smoke权重被明确标记，程序拒绝用其作为正式评测模型；重试时另取输出目录名。

## 3 Qwen训练完成后，自动评测全部70条

下方一条后台任务依次训练Qwen、恢复验证loss最低的checkpoint、保存LoRA、再用LoRA模型完整评测70条。训练失败时不会进入评测。标准输出和错误均追加到同一日志。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup bash -c '
python -u scripts/train_lora.py \
  --config configs/training/qwen_lora.json \
  --out runs/qwen-lora-v1 &&
python -u scripts/experiment.py run-all \
  --config runs/qwen-lora-v1/evaluation-config.json \
  --out runs/qwen-lora-v1-eval
' >> runs/logs/qwen-lora-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/qwen-lora-v1.pid
```

## 4 Qwen完整结束后，再训练并评测LLaVA

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup bash -c '
python -u scripts/train_lora.py \
  --config configs/training/llava_lora.json \
  --out runs/llava-lora-v1 &&
python -u scripts/experiment.py run-all \
  --config runs/llava-lora-v1/evaluation-config.json \
  --out runs/llava-lora-v1-eval
' >> runs/logs/llava-lora-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/llava-lora-v1.pid
```

## 5 查看进度与结果

```bash
tail -n 40 runs/logs/qwen-lora-v1.log
ps -p "$(cat runs/logs/qwen-lora-v1.pid)" -o pid,etime,args
cat runs/qwen-lora-v1/run.json
cat runs/qwen-lora-v1-eval/evaluation/summary.md
```

LLaVA替换路径中的qwen为llava。日志会先逐批显示token/mask检查，再显示训练loss和验证eval_loss，最后显示70条评测进度。训练run.json completed只代表训练结束，完整任务还要看`*-eval/run.json`及evaluation/summary.md。

| 输出 | 用途 |
|---|---|
| runs/模型-lora-v1/adapter/ | 最佳LoRA权重、PEFT配置、基础模型/处理器身份与训练元数据 |
| evaluation-config.json | 自动生成，指向基础模型固定revision和本地adapter，无需手填权重路径 |
| supervision-checks.json | 每条样本的输入长度、prompt长度和监督token数，不保存完整推理过程 |
| trainable-parameters.json | 实际语言层LoRA模块与可训练参数数目 |
| checkpoint-*/ | 每轮训练状态，可用于未完成训练的续跑 |
| *-eval/evaluation/ | Accuracy、Macro-F1、混淆矩阵、逐题结果与错误表 |

若只需重跑正式LoRA评测，使用上面的`experiment.py run-all`部分，另取新的`--out`。不要使用旧qwen.locked.json或llava.json代替自动生成的evaluation-config.json，那会运行未微调基础模型。

## 6 配置与续跑

两个JSON位于configs/training/，共用r16、alpha32、dropout0.05、语言注意力q/k/v/o投影、batch1累计8、学习率5e-5、3轮。视觉编码器、多模态投影器与基础语言权重均冻结；Qwen BF16、LLaVA FP16，沿用基础评测的processor和聊天模板。不量化、不全参训练。

按**60条验证集eval_loss最低**选择checkpoint，目前不是按生成分类Macro-F1选优；70条只在训练方案固定后评测。来源标签噪声、训练与评测题材/标注方式差异可能影响提升。当前max_length4096是长度检查上限，超过就报错，绝不截断图片或答案；调高上限前确认显存，并另取新输出目录。

若未完成的训练中断，原目录有完整checkpoint，则可指定实际存在的checkpoint路径续跑。下方checkpoint-38仅为示例，须换成目录中真实存在的名字；completed输出和已生成的最终adapter不可覆盖。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/X-D-Lab/ty/.cache/huggingface \
HF_HUB_OFFLINE=1 \
nohup python -u scripts/train_lora.py \
  --config configs/training/qwen_lora.json \
  --out runs/qwen-lora-v1 \
  --resume-from-checkpoint runs/qwen-lora-v1/checkpoint-38 \
  >> runs/logs/qwen-lora-v1.log 2>&1 < /dev/null &
```

续跑会核对代码、依赖、配置和数据签名。仅支持单进程单卡；断在保存checkpoint之前则需新输出目录重启。续跑结束后单独执行该模型的完整评测命令。

## 接口依据与验证范围

LoRA模块选择、加载与保存依据[PEFT官方LoRA说明](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora)和[0.17.1实现](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py)；训练入口依据[Transformers 4.57.1 Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py)。固定版本用于沿用现有栈，不宣称是最新版。

本地已通过真实300/60数据的两模型check-only、合成数据泄漏/篡改检查、CPU张量的答案mask测试和适配器身份/内容指纹检查。没有下载7B权重或调用收费API，A800上的真实两步训练、完整训练、权重保存后推理仍待服务器实测；当前没有优化分数。
