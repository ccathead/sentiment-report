# API分步图文判断提示词测试

日期：2026-11-01。方法文件：[cot_v1.txt](../prompts/cot_v1.txt)。代码版本以运行生成的run.json为准。

方法采用零样本CoT风格指令：文本初判→图片补充→图文关系检查→整体裁决。没有few-shot示例，最终只输出标签，不记录或验证模型内部推理过程；报告可称“零样本CoT风格的分步图文判断提示”。它与旧evidence提示有内容重叠，新增明确的文本初判和弱文本情绪分支，是否优于旧提示待实测。按用户要求，提示词已翻译为英文，与baseline使用相同语言；结果仍代表整套提示修改效果，不能单独归因于分步推理。

调用仍使用--prompt cot_v1。若已用中文版本启动过任务，保留旧结果，英文版本另用新输出目录和日志名（例如deepseek-all-cot-v1-en），不要在旧任务上--resume；程序会核对提示词哈希。

配置中的模型ID、接口及生成参数逐项沿用已归档的DeepSeek/Kimi基线config.json，包括512输出预算，不同时改变接口参数。DeepSeek旧基线有一条长度耗尽后的空输出，当前预算仍可能触发这一问题；无效输出继续计错。如果后续增大预算，另建配置并用相同预算重新跑baseline，单独记录该改动。

## 启动

在服务器项目根目录、已有API运行环境中执行。.env必须显式通过--env-file .env加载，程序不会自动读取；其中变量名应为DMX_API_KEY，与两份配置中的api_key_env一致。已导出的同名环境变量优先于.env；若之前导出了空值，可先执行unset DMX_API_KEY，再通过.env加载。不要将密钥写入命令或提交Git。API推理不需要选GPU，也不需要模型缓存/离线设置。

以下按用户要求直接完整跑固定评测集datasets/mvsa_eval_v1的70条，完成一个模型的推理和自动评分后再启动下一个。此轮属于固定评测集上的探索比较；若根据结果修改提示，后续不能声称该评测集是未接触的独立测试集。原训练300/验证60条不变，本轮不读取其标签。

```bash
mkdir -p runs/logs
nohup python -u scripts/experiment.py run-all \
  --config configs/dmx-deepseek.json \
  --env-file .env \
  --prompt cot_v1 \
  --out runs/deepseek-all-cot-v1 \
  >> runs/logs/deepseek-all-cot-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/deepseek-all-cot-v1.pid
```

查看进度和评分：

```bash
tail -n 40 runs/logs/deepseek-all-cot-v1.log
cat runs/deepseek-all-cot-v1/evaluation/summary.md
```

确认日志最终state为completed、评分n=70，并检查无效/缺失数量，然后运行Kimi：

```bash
nohup python -u scripts/experiment.py run-all \
  --config configs/dmx-kimi.json \
  --env-file .env \
  --prompt cot_v1 \
  --out runs/kimi-all-cot-v1 \
  >> runs/logs/kimi-all-cot-v1.log 2>&1 < /dev/null &
echo $! > runs/logs/kimi-all-cot-v1.pid
```

```bash
tail -n 40 runs/logs/kimi-all-cot-v1.log
cat runs/kimi-all-cot-v1/evaluation/summary.md
```

结果目录保存配置、提示模板、请求、原始回答、run.json、逐条预测及evaluation评分文件。重点比较Accuracy、Macro-F1、各类召回率及无效输出。已有目录不能直接重跑；新实验换输出目录和日志名。只有代码/提示/配置/数据均未变的中断任务才添加--resume。

## 本地核验和待验证项

只做不调用API的请求预览、配置一致性及格式检查；真实接口连通性、成本、是否完成70条和优化效果均待用户运行回传。LLaVA与Qwen的LoRA微调已完成由用户确认，训练产物及优化分数尚未读取核验。
