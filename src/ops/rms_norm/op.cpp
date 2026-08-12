#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/gpu_ops.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 1,
                   "RMS norm: expected a matrix and a weight vector");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1],
                   "RMS norm: weight shape mismatch");
    CHECK_ARGUMENT(eps >= 0.0f, "RMS norm: epsilon must be non-negative");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS norm: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                             in->shape()[0], in->shape()[1], eps);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                                in->shape()[0], in->shape()[1], eps);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
