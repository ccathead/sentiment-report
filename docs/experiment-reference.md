# 环境与API配置参考

更新日期：2026-10-31。本页只负责首次安装、模型缓存和API接入。模型运行分别使用[基础评测指南](experiment-guide.md)与[PEFT训练指南](peft-guide.md)，当前不需要开发集、挑战集或freeze操作。

## 1 首次安装评测环境

已安装并跑通sentiment-report环境时，直接激活即可，不重复创建。

```bash
cd /data/X-D-Lab/ty/sentiment-report
conda env create -f environment.yml
conda activate sentiment-report
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -m pip check
```

environment.yml使用项目已选定的Conda镜像。PEFT训练请按PEFT指南克隆此环境并安装requirements-training.txt，无需LLaMA-Factory。只使用API或评分时，安装requirements.txt即可，不需要GPU依赖。

## 2 数据与GPU

当前数据组织：

| 目录 | 用途 |
|---|---|
| datasets/mvsa_sft_v1/ | 训练300、验证60；来源一致性派生标签，未经人工联合复核 |
| datasets/mvsa_eval_v1/ | 固定70条人工复核评测数据 |
| datasets/mvsa_daily_v1/ | 原始冻结文件，仅供汇总或追溯历史结果 |

数据不在Git中，需要完整复制到服务器，包括图片与清单。若服务器只有原始冻结目录，则首次生成单一评测集：

```bash
python scripts/consolidate_evaluation.py
python scripts/experiment.py check-data
```

评测目录已存在时不重复汇总。训练数据另从本地复制，检查命令见PEFT指南。

```bash
nvidia-smi -i 2 --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv
```

各启动命令直接带CUDA_DEVICE_ORDER=PCI_BUS_ID和CUDA_VISIBLE_DEVICES=2。此时模型配置继续使用cuda:0，因为物理GPU2在进程内编号为0。运行前按实际空闲资源确认；显存空闲不代表计算空闲。

## 3 模型缓存

统一缓存目录为/data/X-D-Lab/ty/.cache/huggingface。已完整下载时，沿用运行指南中的HF_HUB_OFFLINE=1，不重复下载。首次下载可执行：

```bash
HF_HUB_OFFLINE=0 hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --cache-dir /data/X-D-Lab/ty/.cache/huggingface/hub

HF_HUB_OFFLINE=0 hf download llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --cache-dir /data/X-D-Lab/ty/.cache/huggingface/hub
```

hf来自已安装的huggingface-hub。下载占磁盘空间；加载和训练占GPU显存。下载慢或日志不更新时先检查进程和缓存文件，不删除已下载权重。

## 4 DMXAPI接入

在项目根目录将.env.example复制为.env，在编辑器填写DMX_API_KEY，文件保留本地，不提交Git。

```bash
[ -f .env ] || cp .env.example .env
python scripts/experiment.py models --env-file .env --out runs/dmx-models.json
[ -f configs/dmx.local.json ] || cp configs/dmx-chat.example.json configs/dmx.local.json
```

models只查询账户可见型号，不执行推理。模型列表输出文件已存在时另取名字。编辑configs/dmx.local.json：将model替换为账户支持图片输入且已确认的型号。此前已运行的接口名称为deepseek-v4.1-flash与kimi-k3，继续使用前以实际账户可用性为准。Chat接口使用chat/completions；若账户使用Responses，则复制dmx-responses.example.json并沿用其接口设置。密钥通过环境变量或--env-file加载，不写进JSON配置。

确认配置与账户费用后，完整评测70条的命令如下；此命令会执行付费API请求，不需要GPU：

```bash
mkdir -p runs/logs
nohup python -u scripts/experiment.py run-all \
  --config configs/dmx.local.json \
  --env-file .env \
  --out runs/dmx-all-baseline-v2 \
  >> runs/logs/dmx-all-baseline-v2.log 2>&1 < /dev/null &
echo $! > runs/logs/dmx-all-baseline-v2.pid
```

切换模型时分别保存本地配置，并更换结果目录、日志和PID名称。temperature=null表示不发送该参数；Chat型号若要求max_tokens，则设置chat_token_parameter=max_tokens。推理参数只按实际接口支持情况填写，不静默丢弃报错参数。DeepSeek原512-token预算曾导致空答案，预算调整作为单独工程修复记录，与提示词改进分开比较。

## 5 输出与核验

run.json记录代码、配置、环境、数据及提示指纹和运行状态；predictions.jsonl保留原始答案。run-all成功后自动生成evaluation/中的metrics.json、summary.md、per_sample.csv、errors.csv及confusion_matrix.csv。无效答案、API错误、缺失预测均计错，不删除失败样本。

原有结果绑定旧数据manifest；复算历史结果时显式传--dataset datasets/mvsa_daily_v1。当前新结果使用datasets/mvsa_eval_v1。代码、配置或数据改变后另取输出目录，保留旧结果。

接口与环境依据：[Conda环境管理](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)、[PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)、[DMX Chat本地图片](https://doc.dmxapi.cn/base64-image.html)、[DMX Responses本地图片](https://doc.dmxapi.cn/res-base64-image.html)。这些是既有实现参考，服务器安装和API是否成功仍以实际运行日志为准。
