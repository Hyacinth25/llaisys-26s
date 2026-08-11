# LLAISYS 推理作业学习手册

> 本文以 [`README_ZN.md`](README_ZN.md) 的作业顺序为主线，结合当前 `inference-homework` 分支中的实际代码，解释每一步解决了什么问题、涉及哪些概念、代码如何组织、怎样验证，以及哪些部分尚未完成。
>
> 状态标记：**（已完成）**、**（部分完成）**、**（还未完成）**。

## 1. 当前完成情况总览

| 阶段 | 当前状态 | 已验证内容 | 尚缺内容 |
|---|---|---|---|
| 作业 0：环境与 Runtime | **已完成（CPU）** | Windows 本地构建、Python 包导入、CPU Runtime 测试 | 完整 1.5B 模型下载与 PyTorch 基线仍未在本地跑完 |
| 作业 1：Tensor | **已完成（必做部分）** | `load`、`isContiguous`、`view`、`permute`、`slice` | `contiguous`、`reshape`、`to` 是进阶功能，**（还未完成）** |
| 作业 2：CPU 算子 | **已完成** | Add 及 7 个指定算子的 F32/F16/BF16 测试 | `rearrange` **（还未完成）**，但不属于 README 指定的 7 个必做算子 |
| 作业 3：Qwen2 推理 | **部分完成** | C++ 模型、权重加载、greedy argmax、动态 KV Cache；微型 Qwen2 的 F32/BF16 与 Transformers 对齐 | DeepSeek-R1-Distill-Qwen-1.5B 真实权重端到端测试 **（还未完成）** |
| 作业 4：CUDA/类 CUDA | **还未完成** | 已保留设备抽象与编译开关 | NVIDIA Runtime、GPU 算子、GPU 模型推理、第二个国产平台全部 **（还未完成）** |
| 提交 | **部分完成** | `inference-homework` 已推送到个人 Fork | 官方 PR、CI 全绿、平台报告 **（还未完成）** |

当前分支的主要提交：

- `9dedd75`：Tensor 元数据变换与数据加载。
- `d5d5a68`：CPU 推理算子。
- `d7aab6f`：Qwen2 推理与 KV Cache。
- `ba391df`：微型 Qwen2 的 BF16 回归测试。

## 2. 先理解 LLAISYS 的整体分层

LLAISYS 不是直接用 Python 写模型，而是采用“Python 易用接口 + C ABI + C++ 后端”的结构：

```text
Python 测试 / 用户代码
        ↓
python/llaisys：Python 风格包装
        ↓
python/llaisys/libllaisys：ctypes 声明
        ↓
include/llaisys：稳定的 C API
        ↓
src/llaisys：C API 到 C++ 对象的桥接
        ↓
src/tensor、src/ops、src/models：真正的计算逻辑
        ↓
src/core、src/device：内存、Runtime 和具体硬件
```

### 2.1 为什么中间要有 C API

C++ 支持类、模板和重载，但不同编译器产生的 C++ ABI 不一定一致。C ABI 更稳定，也容易被 Python 的 `ctypes` 调用。因此项目在 C++ 内部使用 `std::shared_ptr<Tensor>` 等高级结构，对外只暴露不透明指针和基础类型。

例如 Python 调用算子的链路是：

1. `llaisys.Ops.linear(...)` 位于 `python/llaisys/ops.py`。
2. 它调用共享库中的 `llaisysLinear`。
3. `llaisysLinear` 在 `src/llaisys/ops.cc` 中把 C 句柄还原为 C++ Tensor。
4. `src/ops/linear/op.cpp` 做设备、shape、dtype 检查。
5. CPU 张量最终进入 `src/ops/linear/cpu/linear_cpu.cpp`。

这也是后续接入 CUDA 时的重要设计：Python API 不需要改变，只替换底层设备分发。

### 2.2 主要目录的职责

- `include/`：共享库对外的 C API 声明。
- `src/llaisys/`：C API 的实现边界，负责包装和解包 C++ 对象。
- `src/core/`：Context、Runtime、Storage、Allocator 等系统基础设施。
- `src/device/`：CPU、NVIDIA 等设备的 Runtime API。
- `src/tensor/`：Tensor 元数据和存储视图。
- `src/ops/`：算子检查、设备分发和不同设备实现。
- `src/models/`：Qwen2 等完整模型的执行图。
- `python/llaisys/libllaisys/`：用 `ctypes` 描述 C 函数的参数和返回值。
- `python/llaisys/`：面向使用者的 Python 类。
- `test/`：以 PyTorch/Transformers 作为参考答案的测试。

## 3. 作业 0：环境、Fork、构建和 Runtime

## 3.1 Fork 与 Git 远程仓库 **（已完成）**

Fork 的目的不是单纯复制代码，而是获得一个自己有写权限的远程仓库，同时保留与官方仓库同步的能力。

当前配置：

```text
origin   = https://github.com/Hyacinth25/llaisys-26s.git
upstream = https://github.com/wooway777/llaisys-26s.git
branch   = inference-homework
```

三个名称的含义：

- `origin`：自己的 Fork，用于推送工作分支。
- `upstream`：课程官方仓库，用于获取官方更新。
- `inference-homework`：本次推理作业分支，避免直接污染 `main`。

常用命令：

```bash
git fetch upstream
git switch inference-homework
git push origin inference-homework
```

如果官方仓库更新，通常先 `fetch upstream`，再把 `upstream/main` 的变更合入自己的工作分支。临近截止时不要盲目合并大更新，应先确认是否会破坏已经通过的测试。

## 3.2 Xmake 构建系统 **（已完成，CPU）**

`xmake.lua` 把后端拆成多个静态库，最后链接成一个共享库：

```text
llaisys-utils
    ↓
llaisys-device-cpu → llaisys-device
    ↓
llaisys-core
    ↓
llaisys-tensor
    ↓
llaisys-ops-cpu → llaisys-ops
    ↓
llaisys-models
    ↓
llaisys.dll / libllaisys.so
```

这里的“依赖”不仅决定链接顺序，也表达模块边界。例如模型可以依赖算子，但算子不应该反过来依赖某个具体模型。

我们新增了 `llaisys-models` target：

```lua
target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-ops")
    add_files("src/models/*/*.cpp")
target_end()
```

最终 `llaisys` target 生成 DLL/SO，并在 `xmake install` 后复制到 Python 包的 `libllaisys` 目录，供 `ctypes.CDLL` 加载。

本地标准构建流程：

```bash
xmake
xmake install
pip install ./python/
```

本机因为网络和 Windows MinGW 安装路径问题，使用了项目目录内的 Xmake 和 Python 虚拟环境，并手动复制 DLL 完成验证。这是本地环境绕行方式，不是提交代码必须依赖的流程。

## 3.3 Runtime API 是什么 **（CPU 已完成，CUDA 还未完成）**

Runtime API 是框架对设备能力的统一抽象，包含：

- 查询设备数量。
- 设置当前设备。
- 创建和销毁 stream。
- 同步设备或 stream。
- 分配/释放设备内存和主机内存。
- 同步/异步内存复制。

`src/device/cpu/cpu_runtime_api.cpp` 中，CPU 只有一个设备，stream 是空指针，内存分配使用 `malloc/free`，复制使用 `memcpy`。这些函数虽然简单，但接口形式与 CUDA 保持一致。

例如 CPU 的 `memcpySync` 不关心 H2D、D2H 还是 D2D：

```cpp
void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
}
```

对 CPU 而言，所谓“device memory”仍然是普通内存；对 CUDA 而言，同一个接口以后需要映射为 `cudaMemcpy`。

### Context、Runtime、Storage 的关系

- `Context`：线程局部对象，记录该线程当前使用哪个设备 Runtime。
- `Runtime`：管理某个具体设备，例如 `NVIDIA:0`，内部持有 Runtime API、Allocator 和 stream。
- `Storage`：一块实际内存，知道自己属于哪个设备和 Runtime。
- `Tensor`：持有 `Storage` 的共享指针，再叠加 shape、stride 和 offset。

`context()` 使用 `thread_local`：

```cpp
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}
```

这意味着不同线程可以维护各自当前设备，不会因为另一个线程调用 `setDevice` 而互相覆盖。

CPU Runtime 测试命令：

```bash
python test/test_runtime.py --device cpu
```

该测试已经通过。

## 3.4 下载真实模型 **（还未完成）**

README 指定模型为 DeepSeek-R1-Distill-Qwen-1.5B。完整验证需要：

1. 下载 `config.json`、tokenizer 文件和 `model.safetensors`。
2. 用 Transformers 得到参考 token。
3. 释放 Transformers 模型，降低内存占用。
4. 用 LLAISYS 加载同一份权重。
5. 在 `--test` 模式下比较完整 token 序列。

命令：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --test
```

真实 1.5B 权重的这一步目前 **（还未完成）**，应在已申请的算力机完成。

## 4. 作业 1：Tensor

## 4.1 Tensor 不等于一块内存

当前 Tensor 由三部分组成：

```cpp
TensorMeta _meta;          // dtype、shape、strides
core::storage_t _storage;  // 共享的实际内存
size_t _offset;            // 当前视图相对 Storage 的字节偏移
```

这种设计允许多个 Tensor 共享同一块 Storage。例如 `slice`、`permute` 和 `view` 都只创建新的元数据对象，不复制底层数据。

### shape、stride 和 offset

假设一个连续张量形状为 `[2, 3, 5]`，按行优先存储，它的 stride 是 `[15, 5, 1]`。

元素 `[i, j, k]` 的元素偏移为：

$$offset_{element}=i\times15+j\times5+k\times1$$

实际字节地址还要乘以元素大小并加上 Tensor 的 `_offset`：

$$address=storage+\_offset+offset_{element}\times elementSize$$

这解释了为什么转置不一定需要搬数据：只要交换 shape 和 stride，就可以用不同的索引方式解释同一块内存。

## 4.2 `load`：主机数据复制到 Tensor **（已完成）**

实现位置：`src/tensor/tensor.cpp`。

主要步骤：

1. 检查来源指针不为空。
2. 要求目标 Tensor 连续，因为一次性 memcpy 无法正确填充带空洞的视图。
3. 切换到 Tensor 所属设备。
4. 调用统一 Runtime API，复制方向为 `LLAISYS_MEMCPY_H2D`。

核心代码：

```cpp
core::context().setDevice(deviceType(), deviceId());
core::context().runtime().api()->memcpy_sync(
    data(), src_, numel() * elementSize(), LLAISYS_MEMCPY_H2D);
```

这里没有直接调用 `std::memcpy`，因为同一段 Tensor 代码以后也要支持 GPU。

## 4.3 `isContiguous`：判断内存是否连续 **（已完成）**

连续布局从最后一维开始检查：最后一维期望 stride 为 1，再依次乘上后一维长度。

```cpp
ptrdiff_t expected_stride = 1;
for (size_t i = ndim(); i > 0; --i) {
    const size_t dim = i - 1;
    if (_meta.shape[dim] != 1
        && _meta.strides[dim] != expected_stride) {
        return false;
    }
    expected_stride *= _meta.shape[dim];
}
```

长度为 1 的维度不影响实际元素地址，因此代码允许它拥有特殊 stride。这比单纯比较“标准 stride 数组”更准确。

## 4.4 `permute`：只改变维度解释 **（已完成）**

`permute({2, 0, 1})` 的含义是：新张量第 0 维来自旧张量第 2 维，新第 1 维来自旧第 0 维，依此类推。

实现做了三类校验：

- order 长度必须等于 ndim。
- 每个维度编号必须在范围内。
- 每个维度只能出现一次。

然后同时重排 shape 和 stride，并共享原 Storage 与 offset。数据完全没有移动，因此它是 $O(ndim)$ 的元数据操作。

## 4.5 `slice`：调整 shape 与 byte offset **（已完成）**

沿维度 `dim` 取 `[start, end)`：

```cpp
meta.shape[dim] = end - start;
const size_t byte_offset = strides[dim] * start * elementSize();
```

新 Tensor 仍共享 Storage，但 `_offset` 增加 `byte_offset`。

例如 `[3, 4]` 连续张量的 stride 是 `[4, 1]`。沿第 0 维从 1 开始切片，起点向后移动 `1 × 4` 个元素；沿第 1 维从 1 开始，则只移动 1 个元素。

## 4.6 `view`：不复制数据的 reshape **（已完成）**

`view` 首先要求新旧元素总数相同。但元素总数相同仍然不保证可以无复制变形。

例如一个 shape `[2, 3, 5]`、stride `[30, 10, 1]` 的张量，在第二维到第三维之间有内存空洞。把后两维合并成 15 会错误地假设这 15 个元素连续，因此必须拒绝。

当前实现把旧张量划分为若干“连续块”，只允许在块内部拆分或合并维度。判断相邻维度能否属于同一块的核心关系是：

$$stride_{previous}=numel_{current\ chunk}\times baseStride$$

如果关系不成立，就到达一个 chunk boundary。新 shape 必须能够按相同元素数量逐块匹配，否则抛出 `view shape is incompatible with tensor strides`。

`view` 与 `reshape` 的典型区别：

- `view`：必须共享原内存，不允许隐式复制。
- `reshape`：通常先尝试 view；不兼容时可创建连续副本。

本项目中的 `reshape` **（还未完成）**。

## 4.7 Tensor 进阶功能 **（还未完成）**

- `contiguous()`：把任意正 stride Tensor 按逻辑顺序复制到连续内存。
- `reshape()`：能 view 时共享内存，不能 view 时先 contiguous 再 view。
- `to(device)`：跨 CPU/GPU 创建设备副本并执行正确方向的内存复制。
- 负 stride：测试辅助代码明确标注暂不支持。

Tensor 必做测试：

```bash
python test/test_tensor.py
```

当前必做测试已经通过。

## 5. 作业 2：CPU 算子

## 5.1 一个算子的标准组织方式

以 Linear 为例：

```text
src/ops/linear/op.hpp              对内函数声明
src/ops/linear/op.cpp              参数检查与设备分发
src/ops/linear/cpu/linear_cpu.hpp  CPU 实现声明
src/ops/linear/cpu/linear_cpu.cpp  CPU 计算循环
```

`op.cpp` 负责检查：

- Tensor 是否在同一设备。
- shape 是否满足公式。
- dtype 是否一致。
- Tensor 是否连续。

CPU 文件只处理已经验证过的裸指针和尺寸。这样可以避免每种设备重复写相同的参数检查。

## 5.2 F32、F16 和 BF16

三种类型的核心差别：

| 类型 | 大小 | 特点 |
|---|---:|---|
| Float32 | 4 字节 | 精度和动态范围都较高，CPU 计算最直接 |
| Float16 | 2 字节 | 尾数较多、指数较少，精度尚可但容易上溢/下溢 |
| BFloat16 | 2 字节 | 与 F32 接近的指数范围，但尾数较少，常用于大模型 |

当前 CPU 算子的通用策略是：

1. 从 F16/BF16 读取时转为 F32。
2. 中间累加和非线性函数在 F32 中计算。
3. 最终结果再转回原 dtype。

这能显著降低半精度累加误差。`src/utils/types.hpp` 中的 `utils::cast<T>` 统一处理转换。

## 5.3 Argmax **（已完成）**

作用：找到一维 logits 中最大的值和索引。模型 greedy 解码最后就依赖它。

实现从第 0 个元素开始保存 `best`，之后只在 `value > best_value` 时更新。相等时保留更早出现的索引，与常见 argmax 语义一致。

输出约束：

- `max_idx` 必须是 I64。
- `max_val` 与输入 dtype 相同。
- 输入不能为空且暂时只支持一维。

测试：

```bash
python test/ops/argmax.py --device cpu
```

## 5.4 Embedding **（已完成）**

Embedding 可以理解为查表。输入 token id 是行号，权重矩阵每一行是一枚 token 的向量。

如果：

```text
index  = [3, 8]
weight = [vocab_size, hidden_size]
```

输出就是 `weight[3]` 和 `weight[8]` 两行。

因为整行数据不需要做数学运算，CPU 实现直接使用 `memcpy`：

```cpp
std::memcpy(out + i * row_bytes,
            weight + indices[i] * row_bytes,
            row_bytes);
```

实现还检查 token id 是否在 `[0, vocab_size)` 范围内。

## 5.5 Linear **（已完成）**

公式：

$$Y=XW^T+b$$

PyTorch 的 Linear 权重布局是 `[out_features, in_features]`。因此输出 `Y[row, col]` 要与 `weight[col, :]` 做点积，而不是直接按普通的 `XW` 去理解。

核心循环：

```cpp
const T *x = in + row * in_features;
const T *w = weight + col * in_features;
float sum = bias ? cast<float>(bias[col]) : 0.0f;
for (size_t k = 0; k < in_features; ++k) {
    sum += cast<float>(x[k]) * cast<float>(w[k]);
}
```

为了让模型 decode 的单 token Linear 不退化成单线程，代码把所有 `(row, col)` 输出展平为 `output_index`，按 CPU 硬件线程数分块。这样即使 `rows=1`，仍可以并行计算不同输出通道。

局限：这是教学用朴素矩阵乘，并不是高性能 GEMM。正式推理通常使用 MKL、OpenBLAS、oneDNN 或 GPU 上的 cuBLAS。

## 5.6 RMSNorm **（已完成）**

RMSNorm 对每个 token 的 hidden vector 独立计算：

$$rms(x)=\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}$$

$$y_i=\frac{x_i}{rms(x)}w_i$$

它与 LayerNorm 的差别是：RMSNorm 不减均值，只根据均方根缩放。计算更简单，也是 Qwen2 使用的归一化方式。

当前实现先以 F32 累加 `square_sum`，计算共同的 `scale`，再逐元素乘权重。

## 5.7 RoPE **（已完成）**

Transformer 本身并不知道 token 的先后顺序。RoPE 通过按位置旋转 Q/K 的向量分量，把相对位置信息编码进点积。

代码把 head dimension 分成前后两半 `a` 和 `b`，角度为：

$$\phi=\frac{position}{\theta^{2i/head\_dim}}$$

再执行二维旋转：

```text
a' = a cos(phi) - b sin(phi)
b' = b cos(phi) + a sin(phi)
```

位置 ID 不是每次从 0 开始。进入增量 decode 后，新 token 的位置应等于当前 KV Cache 长度。这一点在模型代码中通过 `_cache_length + i` 实现。

## 5.8 Self-Attention **（已完成，CPU）**

基本过程：

1. 计算 $QK^T$。
2. 乘以 $1/\sqrt{head\_dim}$，防止点积随维度增大而过大。
3. 应用 causal mask，禁止 token 看到未来位置。
4. 对可见 key 做 softmax。
5. 用注意力权重对 V 加权求和。

### GQA：为什么 Q 头和 KV 头数量不同

DeepSeek-R1-Distill-Qwen-1.5B 使用多个 Q head 共享较少的 K/V head，这叫 Grouped Query Attention（GQA）。

代码计算：

```cpp
heads_per_group = query_heads / key_value_heads;
key_value_head = query_head / heads_per_group;
```

因此同一组 Q head 会读取同一个 KV head，从而减小 KV Cache。

### 右对齐 causal mask

prefill 时 `query_length == key_length`，第 $i$ 个 query 只能看 `[0, i]`。

decode 时，K/V 中还包含历史缓存，通常 `key_length > query_length`。当前可见 key 数量是：

```cpp
visible_keys = key_length - query_length + query_pos + 1;
```

这相当于 PyTorch `tril(diagonal=key_length-query_length)` 的右对齐掩码。

### 稳定 softmax

直接计算 `exp(score)` 可能溢出。实现先减去最大 score：

$$softmax(x_i)=\frac{e^{x_i-max(x)}}{\sum_j e^{x_j-max(x)}}$$

减去同一个常数不改变 softmax 结果，却显著提高数值稳定性。

## 5.9 SwiGLU **（已完成）**

Qwen2 的 MLP 不是普通 ReLU，而是：

$$SwiGLU(gate,up)=up\odot SiLU(gate)$$

$$SiLU(x)=\frac{x}{1+e^{-x}}$$

模型中会分别做 `gate_proj` 和 `up_proj`，再用 SwiGLU 融合，最后经过 `down_proj` 回到 hidden size。

## 5.10 全部算子测试 **（已完成）**

README 提到 `test/test_ops.py`，但当前仓库没有这个聚合脚本，因此实际逐个运行：

```bash
python test/ops/add.py --device cpu
python test/ops/argmax.py --device cpu
python test/ops/embedding.py --device cpu
python test/ops/linear.py --device cpu
python test/ops/rms_norm.py --device cpu
python test/ops/rope.py --device cpu
python test/ops/self_attention.py --device cpu
python test/ops/swiglu.py --device cpu
```

这些测试均已通过 F32、F16 和 BF16 用例。

`src/ops/rearrange/op.cpp` 仍是 **（还未完成）**，但它不在 README 作业 2 指定的七个算子内。

## 6. 作业 3：Qwen2 大模型推理

## 6.1 当前目标模型结构

真实模型配置的关键参数：

| 参数 | 含义 | 值 |
|---|---|---:|
| `nlayer` | Transformer 层数 | 28 |
| `hs` | hidden size | 1536 |
| `nh` | Query head 数 | 12 |
| `nkvh` | KV head 数 | 2 |
| `dh` | 每个 head 的维度 | 128 |
| `di` | MLP intermediate size | 8960 |
| `voc` | 词表大小 | 151936 |
| dtype | 权重类型 | BF16 |
| `theta` | RoPE 基数 | 10000 |
| `epsilon` | RMSNorm epsilon | $10^{-6}$ |

每层主要权重：

- `input_layernorm.weight`
- `q_proj/k_proj/v_proj` 的 weight，Q/K/V 还包含 bias
- `o_proj.weight`
- `post_attention_layernorm.weight`
- `gate_proj/up_proj/down_proj.weight`

模型外还有 token embedding、最终 RMSNorm 和 `lm_head`。

## 6.2 模型 C API **（已完成）**

`include/llaisys/models/qwen2.h` 暴露了：

- `llaisysQwen2ModelCreate`：根据模型配置分配对象和权重 Tensor。
- `llaisysQwen2ModelDestroy`：释放模型和 C 句柄。
- `llaisysQwen2ModelLoadWeight`：按名称加载一块权重。
- `llaisysQwen2ModelResetCache`：开始新请求前清空逻辑缓存长度。
- `llaisysQwen2ModelInfer`：输入一组 token，返回下一个 greedy token。

Python 只能看到 `LlaisysQwen2Model *` 这个不透明指针，不需要知道内部 C++ 类布局。这是典型的 opaque handle 设计。

`python/llaisys/libllaisys/qwen2.py` 用 `ctypes.Structure` 复刻 `LlaisysQwen2Meta` 的字段顺序。字段类型和顺序必须与 C 结构完全一致，否则会出现 ABI 错位。

## 6.3 创建模型与权重映射 **（已完成）**

`Qwen2Model` 构造函数根据 meta 创建每个权重 Tensor，并建立：

```cpp
std::unordered_map<std::string, tensor_t> _weights_by_name;
```

例如 safetensors 名字：

```text
model.layers.7.self_attn.q_proj.weight
```

会映射到：

```cpp
layers[7].attn_q_w
```

这种按名字映射的好处是 Python 不需要理解 C++ 权重数组布局，只需把 safetensors 中的原始名字传给后端。

`loadWeight` 还会检查 `nbytes` 是否等于 `numel × elementSize`，可以提前发现 config、shape 或 dtype 不匹配。

## 6.4 Safetensors 与 mmap 权重加载 **（已完成）**

Safetensors 文件布局可以简化为：

```text
前 8 字节：JSON header 长度（little-endian uint64）
JSON header：每个 tensor 的 dtype、shape、data_offsets
data region：连续存放的原始 tensor 字节
```

Python 加载器没有用 PyTorch 做模型计算，也没有把整个 3 GB 文件读成 Python bytes。它使用 `mmap.ACCESS_COPY` 映射文件：

```python
header_size = struct.unpack_from("<Q", mapped, 0)[0]
header_end = 8 + header_size
base_address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
```

然后根据 `data_offsets` 计算某个权重的地址，并把地址传给 C++。C++ 的 `Tensor::load` 会立即复制到模型 Storage，因此处理完文件后可以安全关闭 mmap。

这样做解决了两个问题：

- NumPy 原生缺少稳定统一的 BF16 dtype 表达。
- 避免 Python 再持有一份完整权重副本导致内存翻倍。

## 6.5 一层 Qwen2 的执行顺序 **（已完成）**

`src/models/qwen2/model.cpp` 中每一层执行：

```text
hidden
  ├─ RMSNorm
  ├─ Q/K/V Linear
  ├─ Q/K RoPE
  ├─ K/V 写入 KV Cache
  ├─ Self-Attention
  ├─ Output Linear
  └─ Residual Add → hidden

hidden
  ├─ RMSNorm
  ├─ Gate Linear ─┐
  ├─ Up Linear ───┼─ SwiGLU
  ├─ Down Linear ←┘
  └─ Residual Add → hidden
```

注意两个残差连接：

$$hidden=hidden+attention\_output$$

$$hidden=hidden+mlp\_output$$

残差让信息和梯度可以跨层传播。虽然这里做的是推理，没有反向传播，但必须严格复现训练时的网络结构。

## 6.6 Prefill 与 Decode

大模型生成分成两个阶段：

### Prefill

第一次把完整 prompt 一次送入模型：

```python
next_token = infer(prompt, len(prompt))
```

这一阶段会为 prompt 中所有 token 计算 Q/K/V，并把每层 K/V 保存到缓存。

### Decode

之后每次只输入刚生成的一个 token：

```python
next_token = infer([previous_token], 1)
```

新 token 只计算自己的 Q/K/V。Attention 的 K/V 则使用“历史 Cache + 当前 K/V”。

如果没有 KV Cache，每生成一个 token 都要重新计算整个历史序列，计算量会随着序列增长反复浪费。

## 6.7 动态 KV Cache **（已完成）**

每层有一个 Key Cache 和一个 Value Cache，逻辑形状为：

```text
[capacity, num_key_value_heads, head_dim]
```

缓存包含两个状态：

- `_cache_length`：已经有效写入多少 token。
- `_cache_capacity`：当前分配的最大 token 数。

容量不足时采用近似倍增：

```cpp
new_capacity = max(required, old_capacity * 2);
```

再通过 D2D memcpy 把旧缓存复制到新 Storage。倍增策略避免每新增一个 token 就重新分配一次，常见于 `std::vector`、动态数组和推理缓存。

写入当前 token 的 K/V 时，通过 `slice(0, cache_length, total_length)` 得到目标视图，再执行连续内存复制。

开始一次新的 `generate` 前调用 `resetCache()`。它把逻辑长度设为 0，但保留已经申请的容量，可以在下一次请求中复用内存。

## 6.8 最终 logits 与 greedy generation **（已完成）**

28 层结束后只取最后一个 token 的 hidden state：

1. 最终 RMSNorm。
2. `lm_head` Linear 得到 `[vocab_size]` logits。
3. Argmax 选出概率排名最高的 token。
4. 如果等于 EOS token，停止生成；否则继续 decode。

当前 `top_k`、`top_p` 和 `temperature` 参数会被忽略，因为 README 的作业验收明确使用 argmax。随机采样 **（还未完成）**，但不影响 `--test` 模式的 greedy 对齐要求。

## 6.9 微型 Qwen2 对照测试 **（已完成）**

`test/test_qwen2_tiny.py` 创建一个两层小模型：

- vocab 32
- hidden size 16
- 4 个 Q head、2 个 KV head
- head dim 4
- intermediate size 32

测试流程：

1. Transformers 随机初始化模型。
2. 保存为 safetensors。
3. Transformers greedy 生成 3 个 token。
4. LLAISYS 加载完全相同的权重并生成 3 个 token。
5. 比较完整 token 列表。
6. 再调用一次 `generate`，验证 Cache 重置。
7. 分别测试 F32 和 BF16。

命令：

```bash
python test/test_qwen2_tiny.py
```

该测试已经通过。它证明模型结构、权重映射、GQA、RoPE、残差、MLP、argmax 和增量 Cache 在小模型上可以与 Transformers 对齐。

## 6.10 真实 1.5B 模型验证 **（还未完成）**

小模型通过不等于真实模型已经验收。真实模型还可能暴露：

- 3 GB 级权重带来的内存问题。
- 28 层累计产生的 BF16 数值误差。
- 长 prompt 下的 Cache 和 causal mask 问题。
- CPU 朴素 Linear 速度过慢。
- Windows/Linux 不同的 mmap、动态库和编译差异。

因此作业 3 当前应标记为 **（部分完成）**。最终标准仍然是：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --test
```

并看到 LLAISYS token 与 Transformers token 完全一致。

## 7. 作业 4：CUDA 与双平台适配 **（还未完成）**

README 要求 NVIDIA、天数智芯、摩尔线程、沐曦中至少两个平台。当前代码只有 CPU 路径可用。

## 7.1 CUDA Runtime API **（还未完成）**

`src/device/nvidia/nvidia_runtime_api.cu` 仍包含多个 `TO_BE_IMPLEMENTED()`。需要把统一接口映射到 CUDA：

| LLAISYS Runtime API | NVIDIA CUDA 对应概念 |
|---|---|
| `get_device_count` | `cudaGetDeviceCount` |
| `set_device` | `cudaSetDevice` |
| `device_synchronize` | `cudaDeviceSynchronize` |
| `create_stream` | `cudaStreamCreate` |
| `destroy_stream` | `cudaStreamDestroy` |
| `stream_synchronize` | `cudaStreamSynchronize` |
| `malloc_device` | `cudaMalloc` |
| `free_device` | `cudaFree` |
| `malloc_host` | `cudaMallocHost` 或 pinned host memory |
| `memcpy_sync` | `cudaMemcpy` |
| `memcpy_async` | `cudaMemcpyAsync` |

还需要补充 `xmake/nvidia.lua`。当前 `xmake.lua` 已有 `--nv-gpu` 开关，但对应文件尚不存在，所以启用开关后不能形成完整 CUDA 构建。

目标测试：

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
python test/test_runtime.py --device nvidia
```

## 7.2 CUDA 算子 **（还未完成）**

每个算子需要增加 `nvidia/` 实现，并在 `op.cpp` 根据 `deviceType()` 分发。

建议实现方式：

- Add、SwiGLU、RMSNorm、RoPE：自定义 CUDA kernel。
- Embedding、Argmax：自定义 kernel 或归约。
- Linear：优先使用 cuBLAS GEMM，而不是自己写朴素矩阵乘。
- Self-Attention：截止时间内可先写正确版本；性能版可考虑 cuBLAS、FlashAttention 或厂商库。

当前 `add/op.cpp` 中 NVIDIA 分支仍是 `TO_BE_IMPLEMENTED()`，其余新增算子目前遇到非 CPU 设备会直接报告 unsupported device。

## 7.3 GPU 模型推理 **（还未完成）**

模型主体已经通过 `Tensor::create(..., _device, _device_id)` 按设备创建 Tensor，并通过统一 D2D/D2H API 操作缓存，这为 GPU 留出了接口。

但要真正运行，还必须完成：

- NVIDIA Runtime API。
- 全部模型所需 GPU 算子。
- GPU 上的权重加载。
- GPU KV Cache 复制与扩容验证。
- GPU 真实模型 token 对齐。

最终命令：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --test \
  --device nvidia
```

## 7.4 第二个国产平台 **（还未完成）**

第二个平台需要使用厂商兼容工具链和 Runtime，例如天数、沐曦或摩尔线程。虽然很多接口与 CUDA 类似，但不能假设代码无需修改。需要在实际算力机确认：

- 编译器命令和 `.cu` 支持方式。
- Runtime API 名称和兼容程度。
- BLAS 库及链接参数。
- device type 枚举与构建 target。
- 测试结果和已知限制。

作业要求的是“至少两个平台实际适配并说明状态”，只写代码但没有平台运行记录通常不足以验收。

## 8. 测试与调试方法

## 8.1 从小到大的测试顺序

推荐顺序：

1. 编译共享库。
2. Runtime 测试。
3. Tensor 测试。
4. 单个算子定向测试。
5. 全部 CPU 算子回归。
6. 微型 Qwen2 F32。
7. 微型 Qwen2 BF16。
8. 真实 1.5B 短 prompt、少量新 token。
9. 真实模型完整 128 token。
10. GPU Runtime、单算子、模型逐层推进。

不要在 Runtime 或单算子尚未正确时直接调完整模型，否则错误来源太多。

## 8.2 常见错误如何定位

### Shape 错误

先打印每个中间 Tensor 的 shape，重点核对：

- Q：`[seq, nh, dh]`
- K/V：`[total_seq, nkvh, dh]`
- Attention：`[seq, nh, dh]`
- MLP：`[seq, di]`
- hidden：`[seq, hs]`

### 第一个 token 就不同

优先检查：

- 权重名字和 shape。
- Linear 是否使用 $W^T$。
- RMSNorm epsilon。
- RoPE 前后半维旋转方式。
- GQA head 映射。
- 最终 lm_head。

### 第一个 token 相同，后续 token 不同

优先检查：

- `_cache_length` 更新时机。
- 新 token 的 position id。
- K/V 追加位置。
- causal mask 是否右对齐。
- 新请求前是否 reset cache。

### F32 正确，BF16 不正确

优先检查：

- safetensors dtype 与 C++ dtype 是否一致。
- 中间累加是否转为 F32。
- 转换函数是否正确舍入。
- 是否错误地把 BF16 当作 IEEE FP16。

## 8.3 `Tensor::debug()` 的用途

`debug()` 会先同步设备，再打印 shape、stride、dtype 和数据。GPU Tensor 会先 D2H 复制到临时 Tensor。

调完整模型时，可在 PyTorch 和 LLAISYS 的相同位置比较前几个元素，例如：

1. embedding 输出。
2. 第一层 RMSNorm。
3. 第一层 Q/K/V projection。
4. RoPE 后的 Q/K。
5. attention 输出。
6. 第一层 MLP 输出。
7. 最终 logits 的前若干值和 argmax。

一旦找到第一个分叉位置，就只调该算子或该层，不必反复跑完整生成。

## 9. 当前实现的工程限制

理解限制与理解已完成内容同样重要：

- CPU Linear 是教学用朴素实现，虽然有多线程，但性能远低于成熟 BLAS。
- Self-Attention 会分配 `scores` 向量，没有做 FlashAttention 式融合。
- 模型 forward 每层会频繁创建临时 Tensor，尚无内存复用计划。
- 当前只支持 greedy argmax，随机采样 **（还未完成）**。
- C API 的 `device_ids` 当前只使用第一个设备，多卡张量并行 **（还未完成）**。
- 完整 1.5B 模型 CPU/BF16 token 对齐 **（还未完成）**。
- CUDA Runtime 和所有 GPU 算子 **（还未完成）**。
- 第二个国产平台 **（还未完成）**。
- `Tensor::contiguous/reshape/to` **（还未完成）**。
- `rearrange` 算子 **（还未完成）**。
- 官方 PR、CI 和平台复现报告 **（还未完成）**。

## 10. 接下来应按什么顺序学习和完成

### 第一步：真实模型 CPU 短测试 **（还未完成）**

先把 `max_steps` 调小到 1～3，确认首 token 和 Cache decode：

```bash
python test/test_infer.py \
  --model /path/to/model \
  --test \
  --max_steps 3
```

如果失败，按“第一处分叉”逐层比较。

### 第二步：完成 NVIDIA Runtime **（还未完成）**

只运行 Runtime 测试，不急着编译完整模型。理解每个 CPU Runtime 函数如何映射到 CUDA。

### 第三步：GPU 算子逐个迁移 **（还未完成）**

建议顺序：Add → Embedding/Argmax → RMSNorm/SwiGLU/RoPE → Linear → Self-Attention。每完成一个就跑对应 `--device nvidia` 测试。

### 第四步：NVIDIA 端到端模型 **（还未完成）**

先 1 token，再 3 token，最后完整 128 token。记录 GPU 型号、驱动、Toolkit、构建命令和结果。

### 第五步：国产平台 **（还未完成）**

先运行平台环境探针，再迁移 Runtime 和算子。不要在不知道工具链版本时凭空写大量兼容代码。

### 第六步：PR 与报告 **（还未完成）**

报告至少应包含：

- 实现了哪些 Tensor、算子、模型和 Cache 功能。
- Windows/Linux、CPU/GPU 环境。
- 完整构建与测试命令。
- 每个平台的通过/失败矩阵。
- 真实模型生成结果。
- 已知限制和后续优化方向。

## 11. 专业名词速查

| 名词 | 简要解释 |
|---|---|
| ABI | 二进制接口约定；决定函数、结构体在编译后二进制层如何交互 |
| ctypes | Python 调用 C 共享库的标准库机制 |
| Runtime | 对某类硬件的设备、内存、stream 和复制操作的统一封装 |
| Context | 当前线程的设备运行上下文，管理多个 Runtime |
| Storage | 一块实际内存，可被多个 Tensor 视图共享 |
| Tensor | Storage 加 dtype、shape、stride、offset 的多维数据视图 |
| Shape | 每个维度的长度 |
| Stride | 某一维索引增加 1 时，内存中跨过多少个元素 |
| Contiguous | 张量元素按标准行优先顺序紧密排列，没有空洞 |
| View | 不复制数据，只改变 shape/stride 的 Tensor 变换 |
| Operator | 对 Tensor 执行的计算，如 Linear、RMSNorm、Attention |
| F16 | IEEE 半精度浮点，2 字节 |
| BF16 | BFloat16，指数范围接近 F32，常用于大模型 |
| Logits | 模型对词表中每个 token 给出的未归一化分数 |
| Argmax | 选择最大分数对应的索引，即 greedy token |
| Embedding | 把 token id 映射到连续向量的查表操作 |
| RMSNorm | 根据均方根缩放 hidden vector 的归一化方法 |
| RoPE | 通过旋转 Q/K 向量编码位置信息 |
| Attention | 用 Q 与 K 的相似度对 V 加权聚合 |
| GQA | 多个 Query head 共享较少的 KV head，节省 Cache |
| Causal Mask | 禁止当前 token 读取未来 token 的掩码 |
| Softmax | 把任意分数变成和为 1 的权重 |
| SwiGLU | `up × SiLU(gate)`，Qwen2 MLP 使用的门控激活 |
| Residual | 将子层输出加回原 hidden state 的残差连接 |
| Prefill | 一次处理完整 prompt 并建立 KV Cache |
| Decode | 每次只处理一个新 token 的增量生成阶段 |
| KV Cache | 缓存历史 token 的 K/V，避免每步重复计算 |
| Safetensors | 带 JSON 元数据、无代码执行风险的张量文件格式 |
| mmap | 把文件映射到虚拟内存，按需读取，减少额外复制 |
| Stream | GPU 上控制操作执行顺序和异步并发的队列 |
| H2D/D2H/D2D | Host-to-Device、Device-to-Host、Device-to-Device 复制 |
| Kernel | 在 GPU 上由大量线程并行执行的函数 |
| GEMM | 通用矩阵乘，Linear 的核心计算 |
| cuBLAS | NVIDIA 提供的高性能 BLAS/GEMM 库 |
| CI | 持续集成；每次 push/PR 自动构建并运行测试 |

## 12. 最终验收清单

- [x] Fork、`origin/upstream` 和工作分支配置完成。
- [x] CPU Runtime 测试通过。
- [x] Tensor 必做功能与测试通过。
- [x] 七个 CPU 推理算子支持 F32/F16/BF16。
- [x] Qwen2 C++ 模型与 Python ctypes 包装完成。
- [x] Safetensors mmap 权重加载完成。
- [x] 动态 KV Cache 完成。
- [x] 微型 Qwen2 的 F32/BF16 token 与 Transformers 对齐。
- [ ] DeepSeek-R1-Distill-Qwen-1.5B CPU 完整对齐。**（还未完成）**
- [ ] NVIDIA Runtime API。**（还未完成）**
- [ ] NVIDIA 全部算子和模型推理。**（还未完成）**
- [ ] 至少一个国产平台适配。**（还未完成）**
- [ ] 双平台复现报告。**（还未完成）**
- [ ] 官方 PR 和 CI 全绿。**（还未完成）**

