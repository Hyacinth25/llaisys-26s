#include "model.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace llaisys::models {

tensor_t Qwen2Model::create(const std::vector<size_t> &shape, llaisysDataType_t dtype) const {
    return Tensor::create(shape, dtype, _device, _device_id);
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    CHECK_ARGUMENT(meta.nlayer > 0 && meta.hs > 0 && meta.nh > 0 && meta.nkvh > 0,
                   "Qwen2: invalid model dimensions");
    CHECK_ARGUMENT(meta.nh * meta.dh == meta.hs,
                   "Qwen2: hidden size must equal attention heads times head dimension");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0,
                   "Qwen2: attention heads must be divisible by key/value heads");

    in_embed = create({meta.voc, meta.hs}, meta.dtype);
    out_embed = create({meta.voc, meta.hs}, meta.dtype);
    out_norm_w = create({meta.hs}, meta.dtype);
    _weights_by_name.emplace("model.embed_tokens.weight", in_embed);
    _weights_by_name.emplace("lm_head.weight", out_embed);
    _weights_by_name.emplace("model.norm.weight", out_norm_w);

    layers.resize(meta.nlayer);
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto &layer = layers[i];
        layer.attn_norm_w = create({meta.hs}, meta.dtype);
        layer.attn_q_w = create({meta.nh * meta.dh, meta.hs}, meta.dtype);
        layer.attn_q_b = create({meta.nh * meta.dh}, meta.dtype);
        layer.attn_k_w = create({meta.nkvh * meta.dh, meta.hs}, meta.dtype);
        layer.attn_k_b = create({meta.nkvh * meta.dh}, meta.dtype);
        layer.attn_v_w = create({meta.nkvh * meta.dh, meta.hs}, meta.dtype);
        layer.attn_v_b = create({meta.nkvh * meta.dh}, meta.dtype);
        layer.attn_o_w = create({meta.hs, meta.nh * meta.dh}, meta.dtype);
        layer.mlp_norm_w = create({meta.hs}, meta.dtype);
        layer.mlp_gate_w = create({meta.di, meta.hs}, meta.dtype);
        layer.mlp_up_w = create({meta.di, meta.hs}, meta.dtype);
        layer.mlp_down_w = create({meta.hs, meta.di}, meta.dtype);

        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        _weights_by_name.emplace(prefix + "input_layernorm.weight", layer.attn_norm_w);
        _weights_by_name.emplace(prefix + "self_attn.q_proj.weight", layer.attn_q_w);
        _weights_by_name.emplace(prefix + "self_attn.q_proj.bias", layer.attn_q_b);
        _weights_by_name.emplace(prefix + "self_attn.k_proj.weight", layer.attn_k_w);
        _weights_by_name.emplace(prefix + "self_attn.k_proj.bias", layer.attn_k_b);
        _weights_by_name.emplace(prefix + "self_attn.v_proj.weight", layer.attn_v_w);
        _weights_by_name.emplace(prefix + "self_attn.v_proj.bias", layer.attn_v_b);
        _weights_by_name.emplace(prefix + "self_attn.o_proj.weight", layer.attn_o_w);
        _weights_by_name.emplace(prefix + "post_attention_layernorm.weight", layer.mlp_norm_w);
        _weights_by_name.emplace(prefix + "mlp.gate_proj.weight", layer.mlp_gate_w);
        _weights_by_name.emplace(prefix + "mlp.up_proj.weight", layer.mlp_up_w);
        _weights_by_name.emplace(prefix + "mlp.down_proj.weight", layer.mlp_down_w);
    }
}

tensor_t Qwen2Model::weight(const std::string &name) const {
    const auto found = _weights_by_name.find(name);
    CHECK_ARGUMENT(found != _weights_by_name.end(), "Qwen2: unknown weight name");
    return found->second;
}

void Qwen2Model::loadWeight(const std::string &name, const void *data, size_t nbytes) {
    auto destination = weight(name);
    const size_t expected = destination->numel() * destination->elementSize();
    if (nbytes != expected) {
        std::ostringstream message;
        message << "Qwen2: weight byte size mismatch for " << name
                << " (expected " << expected << ", got " << nbytes << ")";
        throw std::invalid_argument(message.str());
    }
    destination->load(data);
}

void Qwen2Model::resetCache() {
    _cache_length = 0;
}

void Qwen2Model::ensureCacheCapacity(size_t required) {
    if (required <= _cache_capacity) {
        return;
    }
    CHECK_ARGUMENT(required <= _meta.maxseq, "Qwen2: sequence exceeds model context length");
    size_t new_capacity = std::max<size_t>(16, std::max(required, _cache_capacity * 2));
    new_capacity = std::min(new_capacity, _meta.maxseq);

    std::vector<tensor_t> new_keys;
    std::vector<tensor_t> new_values;
    new_keys.reserve(_meta.nlayer);
    new_values.reserve(_meta.nlayer);
    const size_t cached_bytes = _cache_length * _meta.nkvh * _meta.dh * utils::dsize(_meta.dtype);
    core::context().setDevice(_device, _device_id);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto key = create({new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype);
        auto value = create({new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype);
        if (_cache_length > 0) {
            core::context().runtime().api()->memcpy_sync(
                key->data(), _key_cache[i]->data(), cached_bytes, LLAISYS_MEMCPY_D2D);
            core::context().runtime().api()->memcpy_sync(
                value->data(), _value_cache[i]->data(), cached_bytes, LLAISYS_MEMCPY_D2D);
        }
        new_keys.push_back(std::move(key));
        new_values.push_back(std::move(value));
    }
    _key_cache = std::move(new_keys);
    _value_cache = std::move(new_values);
    _cache_capacity = new_capacity;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "Qwen2: at least one input token is required");
    const size_t total_length = _cache_length + ntoken;
    ensureCacheCapacity(total_length);
    for (size_t i = 0; i < ntoken; ++i) {
        CHECK_ARGUMENT(token_ids[i] >= 0 && static_cast<size_t>(token_ids[i]) < _meta.voc,
                       "Qwen2: token id out of range");
    }

    auto ids = create({ntoken}, LLAISYS_DTYPE_I64);
    ids->load(token_ids);
    auto hidden = create({ntoken, _meta.hs}, _meta.dtype);
    ops::embedding(hidden, ids, in_embed);

    std::vector<int64_t> host_positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        host_positions[i] = static_cast<int64_t>(_cache_length + i);
    }
    auto positions = create({ntoken}, LLAISYS_DTYPE_I64);
    positions->load(host_positions.data());

    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    const size_t kv_width = _meta.nkvh * _meta.dh;
    for (size_t layer_index = 0; layer_index < _meta.nlayer; ++layer_index) {
        const auto &layer = layers[layer_index];
        auto normalized = create({ntoken, _meta.hs}, _meta.dtype);
        ops::rms_norm(normalized, hidden, layer.attn_norm_w, _meta.epsilon);

        auto q = create({ntoken, _meta.hs}, _meta.dtype);
        auto k = create({ntoken, kv_width}, _meta.dtype);
        auto v = create({ntoken, kv_width}, _meta.dtype);
        ops::linear(q, normalized, layer.attn_q_w, layer.attn_q_b);
        ops::linear(k, normalized, layer.attn_k_w, layer.attn_k_b);
        ops::linear(v, normalized, layer.attn_v_w, layer.attn_v_b);

        auto q3 = q->view({ntoken, _meta.nh, _meta.dh});
        auto k3 = k->view({ntoken, _meta.nkvh, _meta.dh});
        auto v3 = v->view({ntoken, _meta.nkvh, _meta.dh});
        auto q_rotated = create({ntoken, _meta.nh, _meta.dh}, _meta.dtype);
        auto k_rotated = create({ntoken, _meta.nkvh, _meta.dh}, _meta.dtype);
        ops::rope(q_rotated, q3, positions, _meta.theta);
        ops::rope(k_rotated, k3, positions, _meta.theta);

        auto key_destination = _key_cache[layer_index]->slice(0, _cache_length, total_length);
        auto value_destination = _value_cache[layer_index]->slice(0, _cache_length, total_length);
        const size_t append_bytes = ntoken * kv_width * utils::dsize(_meta.dtype);
        core::context().setDevice(_device, _device_id);
        core::context().runtime().api()->memcpy_sync(
            key_destination->data(), k_rotated->data(), append_bytes, LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(
            value_destination->data(), v3->data(), append_bytes, LLAISYS_MEMCPY_D2D);

        auto keys = _key_cache[layer_index]->slice(0, 0, total_length);
        auto values = _value_cache[layer_index]->slice(0, 0, total_length);
        auto attention = create({ntoken, _meta.nh, _meta.dh}, _meta.dtype);
        ops::self_attention(attention, q_rotated, keys, values, attention_scale);

        auto attention2 = attention->view({ntoken, _meta.hs});
        auto projected = create({ntoken, _meta.hs}, _meta.dtype);
        ops::linear(projected, attention2, layer.attn_o_w, nullptr);
        ops::add(hidden, hidden, projected);

        ops::rms_norm(normalized, hidden, layer.mlp_norm_w, _meta.epsilon);
        auto gate = create({ntoken, _meta.di}, _meta.dtype);
        auto up = create({ntoken, _meta.di}, _meta.dtype);
        ops::linear(gate, normalized, layer.mlp_gate_w, nullptr);
        ops::linear(up, normalized, layer.mlp_up_w, nullptr);
        ops::swiglu(gate, gate, up);
        auto down = create({ntoken, _meta.hs}, _meta.dtype);
        ops::linear(down, gate, layer.mlp_down_w, nullptr);
        ops::add(hidden, hidden, down);
    }

    _cache_length = total_length;
    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    auto normalized = create({1, _meta.hs}, _meta.dtype);
    ops::rms_norm(normalized, last_hidden, out_norm_w, _meta.epsilon);
    auto logits = create({1, _meta.voc}, _meta.dtype);
    ops::linear(logits, normalized, out_embed, nullptr);
    auto logits_vector = logits->view({_meta.voc});
    auto max_index = create({1}, LLAISYS_DTYPE_I64);
    auto max_value = create({1}, _meta.dtype);
    ops::argmax(max_index, max_value, logits_vector);

    int64_t result = 0;
    core::context().setDevice(_device, _device_id);
    core::context().runtime().api()->memcpy_sync(
        &result, max_index->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    return result;
}

} // namespace llaisys::models
