#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
void linear_outputs_(T *out, const T *in, const T *weight, const T *bias,
                     size_t begin, size_t end, size_t out_features, size_t in_features) {
    for (size_t output_index = begin; output_index < end; ++output_index) {
        const size_t row = output_index / out_features;
        const size_t col = output_index % out_features;
        const T *x = in + row * in_features;
        const T *w = weight + col * in_features;
        float sum = bias == nullptr ? 0.0f : llaisys::utils::cast<float>(bias[col]);
        for (size_t k = 0; k < in_features; ++k) {
            sum += llaisys::utils::cast<float>(x[k]) * llaisys::utils::cast<float>(w[k]);
        }
        out[output_index] = llaisys::utils::cast<T>(sum);
    }
}

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t rows, size_t out_features, size_t in_features) {
    const size_t output_count = rows * out_features;
    const size_t available = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t thread_count = std::min(output_count, available);
    if (thread_count <= 1 || output_count < 64) {
        return linear_outputs_(out, in, weight, bias, 0, output_count, out_features, in_features);
    }

    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t worker = 0; worker < thread_count; ++worker) {
        const size_t begin = output_count * worker / thread_count;
        const size_t end = output_count * (worker + 1) / thread_count;
        workers.emplace_back(linear_outputs_<T>, out, in, weight, bias,
                             begin, end, out_features, in_features);
    }
    for (auto &worker : workers) {
        worker.join();
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t rows, size_t out_features, size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                       rows, out_features, in_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       rows, out_features, in_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       rows, out_features, in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
