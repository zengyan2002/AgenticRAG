# 本地 Embedding 与 Reranker 的 GPU 推理

本项目通过 `BGEM3FlagModel` 生成 Dense/Sparse 向量，通过 `FlagReranker` 评分。两个客户端分别读取设备配置，切换到 CUDA 不改变模型目录、向量维度或知识库索引结构。

## 此机器的环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存。
- 驱动：560.94。
- 当前两个 API 后端使用的解释器：`D:\Python\Python312\python.exe`。
- 对应 CUDA 构建：`torch==2.13.0+cu126`。需要 GPU 版本，只有驱动或安装 CPU 版 torch 并不能启用 GPU。

安装命令（必须使用后端实际使用的解释器）：

```powershell
& D:\Python\Python312\python.exe -m pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

此 Windows wheel 包含推理所需的 CUDA 运行库，本项目不编译自定义 CUDA 算子，无需另外安装完整 CUDA Toolkit。更换硬件或驱动后需要重新确认版本兼容性。

## 项目配置

在 `.env` 中配置：

```dotenv
BGE_DEVICE=cuda:0
BGE_FP16=true
BGE_BATCH_SIZE=8
BGE_RERANKER_DEVICE=cuda:0
BGE_RERANKER_FP16=true
BGE_RERANKER_BATCH_SIZE=8
```

两个模型的目录保持原值。FP16 能降低显存占用，输出与 CPU FP32 可能有少量数值差异；不能把这类差异直接当成检索错误。

`BGE_RERANKER_BATCH_SIZE` 控制模型每次前向推理的批量，`AGENT_RERANK_BATCH_SIZE` 控制复杂问题链路送给客户端的配对数量，两者不是同一层。8GB 显存先使用前向批量 8；增大批量前应检查长文本、并发查询及同时入库时的显存占用。

入库和查询是两个进程，可能各自加载一份 Embedding，查询进程还会加载 Reranker；显存不能只按一个模型估算。模型按首次使用加载，所以刚启动服务时 `nvidia-smi` 没有模型显存并不代表配置无效。

配置修改后要重启已运行的两个 API 后端；设置和模型客户端都在进程中缓存。PyCharm 的解释器应与安装环境一致，切换为项目 `.venv` 时也要在该环境安装 GPU 版 torch。

## 验证

```powershell
& D:\Python\Python312\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& D:\Python\Python312\python.exe scripts/verify_gpu_models.py --stress --output temp_data/gpu_setup/gpu_verified.json
```

验证脚本使用项目真正的模型客户端，检查参数设备为 `cuda:0`、FP16 类型、1024维 Dense 向量、Sparse 权重和重排分数，并在两个模型同时驻留时进行长文本批量测试。只使用合成文本，不调用远程 LLM，不修改知识库。

如果需要 CPU 对照：

```powershell
& D:\Python\Python312\python.exe scripts/verify_gpu_models.py --device cpu --output temp_data/gpu_setup/cpu_baseline.json
```

脚本里的耗时是一次预热后的模型推理样本，不含首次模型加载，也不代表整个问答接口的加速倍数。

### 本机实测（2026-09-22）

实际后端解释器和项目 `.venv` 均已安装 `torch==2.13.0+cu126`。实际后端解释器的测试结果如下：

| 模型 | CPU FP32 | GPU FP16 |
| --- | --- | --- |
| BGE-M3：8 条长文本 | 15.09 秒 | 0.28 秒 |
| BGE-Reranker：8 对长文本 | 13.97 秒 | 0.23 秒 |

测试使用合成文本，按模型当前的 512 token 上限截断，前向批量为 8。两个模型同时驻留时，PyTorch 峰值分配显存约 2250 MiB，峰值预留显存约 2318 MiB；这不是两个 API 服务并发运行时的总显存测量。

原始结果保存在 `temp_data/gpu_setup/cpu_baseline.json` 和 `temp_data/gpu_setup/gpu_verified.json`。配置生效后已重启入库服务（18000）和查询服务（18001），两个健康检查均返回 `status=ok`。

设备配置为 CUDA 而 CUDA 不可用时，模型工厂会明确报错，不会静默转到 CPU。显存不足时可调低两个前向批大小，并减少同时入库和查询的任务数。
