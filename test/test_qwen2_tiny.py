import tempfile
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import llaisys


def run_tiny_qwen2(dtype):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=30,
        eos_token_id=31,
        tie_word_embeddings=False,
        torch_dtype=dtype,
    )
    reference = Qwen2ForCausalLM(config).to(dtype=dtype).eval()
    prompt = [1, 4, 2, 8]

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory)
        reference.save_pretrained(model_path, safe_serialization=True)
        with torch.no_grad():
            expected = reference.generate(
                torch.tensor([prompt]),
                max_new_tokens=3,
                do_sample=False,
            )[0].tolist()

        candidate = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
        actual = candidate.generate(prompt, max_new_tokens=3, top_k=1)
        assert actual == expected, f"LLAISYS={actual}, Transformers={expected}"

        # A second generation verifies that resetCache discards the prior request.
        repeated = candidate.generate(prompt, max_new_tokens=3, top_k=1)
        assert repeated == expected


def test_tiny_qwen2():
    run_tiny_qwen2(torch.float32)
    run_tiny_qwen2(torch.bfloat16)


if __name__ == "__main__":
    test_tiny_qwen2()
    print("\033[92mTiny Qwen2 inference test passed!\033[0m")
