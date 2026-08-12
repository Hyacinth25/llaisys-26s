#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel);
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel);
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t count, size_t vocab_size,
               size_t hidden_size);
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t rows,
            size_t out_features, size_t in_features);
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t hidden_size, float eps);
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t sequence, size_t heads,
          size_t head_dim, float theta);
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t query_length, size_t key_length,
                    size_t query_heads, size_t key_value_heads,
                    size_t head_dim, float scale);
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel);

} // namespace llaisys::ops::nvidia
