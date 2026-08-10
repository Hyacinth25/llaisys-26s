#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t count, size_t vocab_size, size_t hidden_size) {
    const auto *indices = reinterpret_cast<const int64_t *>(index);
    const size_t row_bytes = hidden_size * llaisys::utils::dsize(type);
    for (size_t i = 0; i < count; ++i) {
        CHECK_ARGUMENT(indices[i] >= 0 && static_cast<size_t>(indices[i]) < vocab_size,
                       "Embedding: index out of range");
        std::memcpy(out + i * row_bytes,
                    weight + static_cast<size_t>(indices[i]) * row_bytes,
                    row_bytes);
    }
}
} // namespace llaisys::ops::cpu
