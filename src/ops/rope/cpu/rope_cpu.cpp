#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t sequence, size_t heads, size_t head_dim, float theta) {
    const size_t half = head_dim / 2;
    for (size_t token = 0; token < sequence; ++token) {
        for (size_t i = 0; i < half; ++i) {
            const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
            const float angle = static_cast<float>(pos_ids[token]) / std::pow(theta, exponent);
            const float sin_value = std::sin(angle);
            const float cos_value = std::cos(angle);
            for (size_t head = 0; head < heads; ++head) {
                const size_t base = (token * heads + head) * head_dim;
                const float a = llaisys::utils::cast<float>(in[base + i]);
                const float b = llaisys::utils::cast<float>(in[base + half + i]);
                out[base + i] = llaisys::utils::cast<T>(a * cos_value - b * sin_value);
                out[base + half + i] = llaisys::utils::cast<T>(b * cos_value + a * sin_value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t sequence, size_t heads, size_t head_dim, float theta) {
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     positions, sequence, heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     positions, sequence, heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     positions, sequence, heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
