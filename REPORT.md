# LLAISYS 作业 0–4 简要报告

## 1. 作业内容

本次提交完成了 LLAISYS 作业 0–4：

- 完成 Tensor 的 `load`、`isContiguous`、`view`、`permute` 和 `slice`。
- 完成 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU 的 CPU 实现，支持 F32、F16、BF16。
- 完成 Qwen2 C++ 推理后端、C API、Python ctypes 包装、Safetensors 权重加载、greedy argmax 生成和动态 KV Cache。
- 完成 CUDA-compatible Runtime 与上述 8 个 GPU 算子，并适配天数智芯 CoreX 和 NVIDIA CUDA 两套工具链。

## 2. 复现流程

### 2.1 CPU 构建与测试

在 Windows PowerShell 或 Ubuntu 终端中进入仓库根目录，执行：

```bash
xmake f -c -m release
xmake
xmake install
pip install ./python

python test/test_runtime.py --device cpu
python test/test_tensor.py
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
  python test/ops/$op.py --device cpu
done
python test/test_qwen2_tiny.py
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --test --max_steps 3
```

Windows PowerShell 中可将算子循环替换为：

```powershell
"add", "argmax", "embedding", "linear", "rms_norm", "rope", "self_attention", "swiglu" |
  ForEach-Object { python "test/ops/$_.py" --device cpu }
```

### 2.2 NVIDIA RTX 4090 构建与测试

```bash
export CUDA_HOME=/usr/local/cuda-12.8
xmake f -c -m release --nv-gpu=y
xmake
xmake install
pip install ./python

python test/test_runtime.py --device nvidia
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
  python test/ops/$op.py --device nvidia
done
python -c "import torch, llaisys; from test.test_qwen2_tiny import run_tiny_qwen2; run_tiny_qwen2(torch.float32, llaisys.DeviceType.NVIDIA); run_tiny_qwen2(torch.bfloat16, llaisys.DeviceType.NVIDIA)"
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --test --max_steps 3
```

### 2.3 天数智芯 BI-V150 构建与测试

BI-V150 使用 CoreX 环境。构建脚本检测到 `/usr/local/corex` 后，会自动使用 CoreX Clang 和 `ivcore` 目标；测试命令与 NVIDIA 平台相同，设备参数仍为 `--device nvidia`。

```bash
xmake f -c -m release --nv-gpu=y
xmake
xmake install
pip install ./python

python test/test_runtime.py --device nvidia
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
  python test/ops/$op.py --device nvidia
done
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --test --max_steps 3
```

## 3. 复现结果与平台状态

| 平台 | 软件环境 | 构建与 Runtime | Tensor/算子 | Qwen2 推理 |
|---|---|---:|---:|---:|
| Windows CPU | Windows、xmake | 通过 | Tensor 与 8 个 CPU 算子通过 | tiny Qwen2 F32/BF16 通过 |
| Ubuntu CPU | Ubuntu 24.04、xmake | 通过 | Tensor 与 8 个 CPU 算子通过 | 真实 1.5B BF16 3-token 与 Transformers 对齐 |
| 天数智芯 BI-V150 32 GB | CoreX 4.4.0、`clang++ -x ivcore` | GPU Runtime 通过 | 8 个 GPU 算子 F32/F16/BF16 通过 | tiny F32/BF16 对齐；真实 1.5B BF16 3-token 对齐 |
| NVIDIA RTX 4090 24 GB | Driver 570.124.06、CUDA Toolkit 12.8、`nvcc 12.8` | GPU Runtime 通过 | 8 个 GPU 算子 F32/F16/BF16 通过 | tiny F32/BF16 对齐；真实 1.5B BF16 完整生成对齐 |

真实模型使用 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，Prompt 为 `Who are you?`，BF16 精度，采用 greedy argmax。`--max_steps 3` 短测试中，Transformers 与 LLAISYS 得到相同 token 序列：

```text
[151646, 151646, 151644, 15191, 525, 498, 30,
 151645, 151648, 198, 91786, 0, 358]
```

RTX 4090 上还使用默认 `--max_steps 128` 完成了完整生成测试：模型生成 83 个新 token 后到达 EOS，LLAISYS 与 Transformers 的完整 token 序列一致。
