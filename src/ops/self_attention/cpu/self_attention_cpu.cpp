#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t query_length, size_t key_length,
                     size_t query_heads, size_t key_value_heads,
                     size_t head_dim, float scale) {
    const size_t heads_per_group = query_heads / key_value_heads;
    std::vector<float> scores(key_length);

    for (size_t query_pos = 0; query_pos < query_length; ++query_pos) {
        // PyTorch's tril(diagonal=key_length-query_length) right-aligns the
        // causal mask when cached keys make the key sequence longer than q.
        const size_t visible_keys = key_length - query_length + query_pos + 1;
        for (size_t query_head = 0; query_head < query_heads; ++query_head) {
            const size_t key_value_head = query_head / heads_per_group;
            const size_t query_base = (query_pos * query_heads + query_head) * head_dim;

            float max_score = -INFINITY;
            for (size_t key_pos = 0; key_pos < visible_keys; ++key_pos) {
                const size_t key_base = (key_pos * key_value_heads + key_value_head) * head_dim;
                float dot = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    dot += llaisys::utils::cast<float>(q[query_base + dim])
                         * llaisys::utils::cast<float>(k[key_base + dim]);
                }
                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float denominator = 0.0f;
            for (size_t key_pos = 0; key_pos < visible_keys; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += scores[key_pos];
            }

            const size_t out_base = query_base;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (size_t key_pos = 0; key_pos < visible_keys; ++key_pos) {
                    const size_t value_base = (key_pos * key_value_heads + key_value_head) * head_dim;
                    value += (scores[key_pos] / denominator)
                           * llaisys::utils::cast<float>(v[value_base + dim]);
                }
                out[out_base + dim] = llaisys::utils::cast<T>(value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t query_length, size_t key_length,
                    size_t query_heads, size_t key_value_heads,
                    size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               query_length, key_length, query_heads, key_value_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               query_length, key_length, query_heads, key_value_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               query_length, key_length, query_heads, key_value_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
