#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3
                       && k->ndim() == 3 && v->ndim() == 3,
                   "Self attention: all tensors must be three-dimensional");
    CHECK_SAME_SHAPE(attn_val->shape(), q->shape());
    CHECK_SAME_SHAPE(k->shape(), v->shape());

    const size_t query_length = q->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t head_dim = q->shape()[2];
    const size_t key_length = k->shape()[0];
    const size_t key_value_heads = k->shape()[1];
    CHECK_ARGUMENT(k->shape()[2] == head_dim, "Self attention: head dimensions must match");
    CHECK_ARGUMENT(query_length <= key_length,
                   "Self attention: query length cannot exceed key/value length");
    CHECK_ARGUMENT(key_value_heads > 0 && query_heads % key_value_heads == 0,
                   "Self attention: query heads must be divisible by key/value heads");
    ASSERT(attn_val->isContiguous() && q->isContiguous()
               && k->isContiguous() && v->isContiguous(),
           "Self attention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), query_length, key_length,
                                   query_heads, key_value_heads, head_dim, scale);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
