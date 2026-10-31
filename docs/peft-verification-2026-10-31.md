# PEFT路线实现核验

日期：2026-10-31。用户要求从LLaMA-Factory改为直接Transformers + PEFT，保留已生成的训练300、验证60和固定评测70。

## 改动与依据

新增直接训练入口、两个训练JSON、PEFT依赖、严格多模态答案mask、语言注意力LoRA筛选、最佳验证loss适配器保存及已有HF评测入口的适配器加载。基于官方固定版本接口，沿用原模型revision和处理器；不使用评测标签选训练参数。旧LLaMA-Factory文档与配置标为历史方案。

此时基础Git版本为d3b5d35ca917467d9f5e9e70f89b384b2920f5c5，实现为其后本地改动；确切发布版本以包含本文件的提交为准。训练run.json自动记录实际代码提交、源文件指纹、依赖、数据、配置和随机种子。

## 本地环境与命令

Windows，Python3.12.3，PyTorch2.8.0+cpu，Pillow11.3.0；无CUDA、未安装Transformers/PEFT运行依赖，也没有7B权重。依赖版本接口经官方代码核对；不能将CPU检查写成实际训练栈已实测兼容。

```bash
python -m unittest discover -s tests
python scripts/train_lora.py --config configs/training/qwen_lora.json --check-only
python scripts/train_lora.py --config configs/training/llava_lora.json --check-only
node tests/test_review_page.cjs
python -m compileall -q src scripts/train_lora.py
git diff --check
```

## 结果与边界

- 54项Python测试通过：原推理回归、源标签导出、合成数据跨集合泄漏/篡改拒绝、LoRA语言层筛选、适配器基础身份/内容指纹与smoke拒绝；其中4项用真实CPU张量和模拟processor检查损失mask、前缀不一致、长度超限和错误批量。
- 两模型真实300/60数据check-only通过，来源标签策略为source-agreement。检查读取评测输入和图片做去重，不读取评测gold。未进行人工联合标注或近重复复核。
- 复核HTML动态计数、过滤与草稿保持检查通过。
- 尚未执行：真实PEFT/Trainer前反向、A800显存检查、两个原生processor实际token/mask检查、完整训练、PEFT权重保存后重新加载及70条推理。
- 风险和下一步：服务器克隆原环境并安装固定PEFT后，先运行指南中的两步smoke；模板前缀、监督答案、非有限loss/梯度、超长样本均会报错保留失败run.json。两步检查成功后再完整训练与评测，不预填提升数字。

训练默认r16/alpha32/dropout0.05、学习率5e-5、3轮、batch1累计8、seed20261031，max_length4096只检查不截断。Qwen BF16、LLaVA FP16，验证loss最低选优。数据及基础权重未改变；脚本不提交图片、适配器、日志或密钥。
