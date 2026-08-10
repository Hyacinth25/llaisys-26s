#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::models {

struct Qwen2LayerWeights {
    tensor_t attn_norm_w;
    tensor_t attn_q_w;
    tensor_t attn_q_b;
    tensor_t attn_k_w;
    tensor_t attn_k_b;
    tensor_t attn_v_w;
    tensor_t attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w;
    tensor_t mlp_up_w;
    tensor_t mlp_down_w;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<Qwen2LayerWeights> layers;

    tensor_t weight(const std::string &name) const;
    void loadWeight(const std::string &name, const void *data, size_t nbytes);
    int64_t infer(const int64_t *token_ids, size_t ntoken);
    void resetCache();

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _cache_length{0};
    size_t _cache_capacity{0};
    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    std::unordered_map<std::string, tensor_t> _weights_by_name;

    tensor_t create(const std::vector<size_t> &shape, llaisysDataType_t dtype) const;
    void ensureCacheCapacity(size_t required);
};

} // namespace llaisys::models
