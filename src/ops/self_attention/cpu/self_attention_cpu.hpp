#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t query_length, size_t key_length,
                    size_t query_heads, size_t key_value_heads,
                    size_t head_dim, float scale);
}
