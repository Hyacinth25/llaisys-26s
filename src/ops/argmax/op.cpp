#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/gpu_ops.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be one-dimensional");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals cannot be empty");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1,
                   "Argmax: outputs must each contain one element");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64,
                   "Argmax: index output must use int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           vals->dtype(), vals->numel());
    }
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              vals->dtype(), vals->numel());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
