#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be one-dimensional");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2,
                   "Embedding: weight and output must be two-dimensional");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64,
                   "Embedding: index must use int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0]
                       && out->shape()[1] == weight->shape()[1],
                   "Embedding: output shape mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(),
                              index->numel(), weight->shape()[0], weight->shape()[1]);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
