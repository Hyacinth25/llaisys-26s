#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <memory>
#include <string>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;
    LlaisysQwen2Weights weights{};
    std::vector<llaisysTensor_t> handles;
};

namespace {
llaisysTensor_t wrap(LlaisysQwen2Model *model, const llaisys::tensor_t &tensor) {
    auto handle = new LlaisysTensor{tensor};
    model->handles.push_back(handle);
    return handle;
}

llaisysTensor_t *make_array(size_t size) {
    return new llaisysTensor_t[size]{};
}

void expose_weights(LlaisysQwen2Model *model, size_t nlayer) {
    auto &out = model->weights;
    out.in_embed = wrap(model, model->model->in_embed);
    out.out_embed = wrap(model, model->model->out_embed);
    out.out_norm_w = wrap(model, model->model->out_norm_w);
    out.attn_norm_w = make_array(nlayer);
    out.attn_q_w = make_array(nlayer);
    out.attn_q_b = make_array(nlayer);
    out.attn_k_w = make_array(nlayer);
    out.attn_k_b = make_array(nlayer);
    out.attn_v_w = make_array(nlayer);
    out.attn_v_b = make_array(nlayer);
    out.attn_o_w = make_array(nlayer);
    out.mlp_norm_w = make_array(nlayer);
    out.mlp_gate_w = make_array(nlayer);
    out.mlp_up_w = make_array(nlayer);
    out.mlp_down_w = make_array(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        const auto &layer = model->model->layers[i];
        out.attn_norm_w[i] = wrap(model, layer.attn_norm_w);
        out.attn_q_w[i] = wrap(model, layer.attn_q_w);
        out.attn_q_b[i] = wrap(model, layer.attn_q_b);
        out.attn_k_w[i] = wrap(model, layer.attn_k_w);
        out.attn_k_b[i] = wrap(model, layer.attn_k_b);
        out.attn_v_w[i] = wrap(model, layer.attn_v_w);
        out.attn_v_b[i] = wrap(model, layer.attn_v_b);
        out.attn_o_w[i] = wrap(model, layer.attn_o_w);
        out.mlp_norm_w[i] = wrap(model, layer.mlp_norm_w);
        out.mlp_gate_w[i] = wrap(model, layer.mlp_gate_w);
        out.mlp_up_w[i] = wrap(model, layer.mlp_up_w);
        out.mlp_down_w[i] = wrap(model, layer.mlp_down_w);
    }
}

void destroy_weight_arrays(LlaisysQwen2Weights &weights) {
    delete[] weights.attn_norm_w;
    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;
    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;
    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;
    delete[] weights.attn_o_w;
    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;
}
} // namespace

__C {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                            llaisysDeviceType_t device,
                                            int *device_ids,
                                            int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2: metadata cannot be null");
    CHECK_ARGUMENT(device_ids != nullptr && ndevice > 0, "Qwen2: at least one device is required");
    auto result = std::make_unique<LlaisysQwen2Model>();
    result->model = std::make_unique<llaisys::models::Qwen2Model>(*meta, device, device_ids[0]);
    expose_weights(result.get(), meta->nlayer);
    return result.release();
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }
    destroy_weight_arrays(model->weights);
    for (auto handle : model->handles) {
        delete handle;
    }
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model cannot be null");
    return &model->weights;
}

void llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model,
                                 const char *name,
                                 const void *data,
                                 size_t nbytes) {
    CHECK_ARGUMENT(model != nullptr && name != nullptr, "Qwen2: invalid weight load request");
    model->model->loadWeight(std::string(name), data, nbytes);
}

void llaisysQwen2ModelResetCache(LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model cannot be null");
    model->model->resetCache();
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model,
                               int64_t *token_ids,
                               size_t ntoken) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model cannot be null");
    return model->model->infer(token_ids, ntoken);
}
}
