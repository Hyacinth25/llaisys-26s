#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t rows, size_t hidden_size, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        const T *x = in + row * hidden_size;
        T *y = out + row * hidden_size;
        float square_sum = 0.0f;
        for (size_t col = 0; col < hidden_size; ++col) {
            const float value = llaisys::utils::cast<float>(x[col]);
            square_sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(square_sum / static_cast<float>(hidden_size) + eps);
        for (size_t col = 0; col < hidden_size; ++col) {
            const float value = llaisys::utils::cast<float>(x[col]);
            const float w = llaisys::utils::cast<float>(weight[col]);
            y[col] = llaisys::utils::cast<T>(value * scale * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
