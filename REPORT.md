# LLAISYS 作业简要报告

## 1. 项目内容

本次作业完成了 LLAISYS 作业 0–4 的核心内容：

- 实现 Tensor 的 `load`、`isContiguous`、`view`、`permute` 和 `slice`。
- 实现 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU 的 CPU 版本，支持 F32、F16、BF16。
- 实现 Qwen2 C++ 推理后端、C API 和 Python ctypes 包装。
- 实现 Safetensors 权重加载、greedy argmax 生成和动态 KV Cache。
- 实现 CUDA-compatible Runtime 和上述 8 个 GPU 算子。
- 适配天数智芯 BI-V150/CoreX 与 NVIDIA RTX 4090/CUDA 两个平台。

## 2. 实现思路

### 2.1 Tensor 与算子

Tensor 由 Storage、offset 和 shape/stride 元数据组成。`view`、`permute` 与 `slice` 只创建共享同一 Storage 的新 Tensor，并调整元数据和偏移量，因此不会复制底层数据；`isContiguous` 从最后一个维度向前检查 stride，用于判断目标形状能否安全地进行零拷贝 `view`。`load` 则通过当前设备的 Runtime API 将主机数据复制到 Tensor 所在设备。

CPU 算子按照“公共分发层 + 设备实现层”组织。公共 `op.cpp` 检查设备和参数，再分发到 CPU 或 GPU 实现。F32、F16、BF16 统一先转换为 float 参与中间计算，输出时转换回原 dtype，既复用主要计算逻辑，也减少半精度累加带来的误差。Linear、RMSNorm、RoPE、Self-Attention 等算子均按照 Qwen2 实际张量布局实现，其中 Self-Attention 支持 GQA、因果掩码及历史 KV。

### 2.2 Qwen2 推理

模型后端使用 C++ 实现，通过 C API 导出创建、销毁、加载权重、重置缓存和推理接口，再由 Python ctypes 包装。Python 层读取 `config.json` 构造模型元数据，并使用 mmap 解析 Safetensors 文件头和权重偏移；C++ 后端收到权重后立即复制到目标设备，模型计算本身不依赖 PyTorch 或其他 Python 推理框架。

前向过程按照 Qwen2 结构依次执行 Embedding、每层 Attention/MLP、残差连接、最终 RMSNorm、lm_head 和 Argmax。文本生成分为 Prefill 与 Decode：Prefill 一次处理完整 prompt，Decode 之后每次只输入最新 token。每层保存 K/V Cache，容量不足时按倍增策略扩容并通过设备间复制保留旧数据，从而避免每生成一个 token 都重新计算全部上下文。RoPE 的 position id 和 Attention 的 causal mask 都基于当前缓存长度计算，保证增量推理与 Transformers 的 greedy 结果一致。

### 2.3 CUDA 与双平台适配

GPU 侧通过统一 Runtime API 封装设备选择、显存与页锁定内存分配、H2D/D2H/D2D 复制、Stream 和同步操作。各算子在设备分发层复用同一套接口，权重、临时 Tensor 和 KV Cache 都直接创建在目标 GPU，只有最终 Argmax token 需要复制回主机。

GPU kernel 保持同一份 CUDA-compatible 源码，构建规则根据环境选择工具链：检测到 `/usr/local/corex` 时使用 CoreX Clang 和 `ivcore` 目标，否则使用 `$CUDA_HOME/bin/nvcc` 编译标准 CUDA。这样既保留统一的算子实现，也能够在天数智芯 BI-V150 和 NVIDIA RTX 4090 上分别完成实机编译与验证。

## 3. 复现流程

CPU 构建与测试：

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
python test/test_infer.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --test
```

GPU 构建与测试：

```bash
export CUDA_HOME=/usr/local/cuda-12.8  # NVIDIA 环境
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

## 4. 复现结果

| 平台 | 软件环境 | Runtime | 8 个 GPU 算子 | tiny Qwen2 | 真实 1.5B Qwen2 |
|---|---|---:|---:|---:|---:|
| 天数智芯 BI-V150 32 GB | CoreX 4.4.0 | 通过 | F32/F16/BF16 全部通过 | F32/BF16 token 对齐 | BF16 3-token 对齐 |
| NVIDIA RTX 4090 24 GB | Driver 570.124.06、CUDA 12.8 | 通过 | F32/F16/BF16 全部通过 | F32/BF16 token 对齐 | BF16 完整 token 对齐 |

此外，Windows 和 Ubuntu CPU 环境下的 Runtime、Tensor 与 8 个 CPU 算子均通过；Ubuntu CPU 的真实 1.5B BF16 推理与 Transformers token 序列完全一致。RTX 4090 上按默认 128-token 上限测试时，模型生成 83 个新 token 后到达 EOS，LLAISYS 与 Transformers 的完整 token 序列一致。
