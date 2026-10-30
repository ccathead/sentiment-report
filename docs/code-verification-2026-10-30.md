# 实验代码核验记录

日期：2026-10-30。范围：实验代码首次交付，尚未开展真实模型效果比较。

## 环境与输入

- 本地核验：Windows，Python 3.12.14，Pillow 12.3.0；本机没有 PyTorch。
- 云端目标：Python 3.10，PyTorch 2.6.0 / CUDA 12.4，Transformers 4.57.1；依赖版本在 requirements 文件固定，公开 PyPI 版本存在性已核对。新实验的云端安装与 GPU 推理仍待运行。
- 数据：`mvsa_daily_v1`，dev=12 / test=47 / challenge=11；manifest SHA-256 为 `fb1ef169f7237f5c6ed6039398e3b302390144265231952ce4d402da53cdb8f8`。
- 实验代码提交：`ffdb76550f5315fe0a62df93f7ecd12fdfb9e7ff`。之后实际运行会在 `run.json` 自动记录 commit、源码哈希、环境、配置与命令。

## 已完成核验

| 检查 | 结果 | 能证明的范围 |
|---|---|---|
| `python -m unittest discover -s tests -v` | 24 项通过 | 合成数据和模拟传输的逻辑正确性；不代表真实模型效果 |
| `python scripts/experiment.py check-data` | 155 个冻结文件全部匹配 | 正式数据完整、三组 ID 不交叉、标签与输入 ID 对齐 |
| Qwen / LLaVA / DMX 图文及 Qwen 文字 dry-run | 请求预览生成成功 | 真实数据路径、prompt 展开、输入条件和配置可衔接；无模型预测 |
| HF revision 查询 | 两个指定提交可读取 | 固定版本存在，尚未下载权重 |
| PyPI 固定依赖元数据查询 | 所列版本均存在 | 下载渠道可用，不等同于目标 GPU 环境安装验证 |

关键测试覆盖：仅从输入文件推理；拒绝多余 gold 字段；哈希变化和路径越界报错；文本组不传图片；完整标签严格解析；漏项、报错和截断响应计错；按 ID 对齐；重复/未知 ID 拒绝；429 重试与 401 停止；密钥/图片回显清理；断点续跑跳过已落盘样本；配置变化不能接续旧运行；测试/挑战必须使用冻结协议。

本地首次测试因执行环境默认临时目录不可写而失败；将测试临时目录指定到本项目 `work/test-temp` 后，测试正常通过。没有为此修改数据或放宽检查。该路径仅为本机核验设置，Linux CI 可使用默认临时目录。

## 待验证与下一步

1. 在 4090 D 安装依赖，各跑 3 条开发样本，核对图像张量、输出和显存。
2. 用户本地配置 DMX 令牌，查询账户模型列表，再验证一种闭源视觉模型的实际路由；没有使用密钥或计费推理。
3. 完整开发集选择提示并填写冻结协议，再运行测试与挑战集。
4. 保留实际错误记录及至少两个人工案例分析，不将测试夹具的模拟输出写入课程结果。

GitHub Actions 只执行 CPU 测试，不读取数据、下载模型或访问 DMXAPI。首次提交已在 Ubuntu / Python 3.10 / requirements.txt 固定依赖下通过[远端自动测试](https://github.com/ccathead/sentiment-report/actions/runs/41019019535)，状态为 success。

仓库已确认是 private，远端 main 已收到上述代码提交。本地默认分支 main 已设置跟踪 origin/main。
