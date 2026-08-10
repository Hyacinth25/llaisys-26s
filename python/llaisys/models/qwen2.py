from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType, LlaisysQwen2Meta

import ctypes
import json
import mmap
import struct
from pathlib import Path


_DTYPES = {
    "bfloat16": DataType.BF16,
    "float16": DataType.F16,
    "float32": DataType.F32,
}

_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2 config: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        dtype_name = str(config.get("torch_dtype", config.get("dtype", "bfloat16")))
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported Qwen2 weight dtype: {dtype_name}")

        hidden_size = int(config["hidden_size"])
        attention_heads = int(config["num_attention_heads"])
        self._end_token = int(config["eos_token_id"])
        rope_parameters = config.get("rope_parameters") or {}
        rope_theta = config.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
        meta = LlaisysQwen2Meta(
            dtype=int(_DTYPES[dtype_name]),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=attention_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=int(config.get("head_dim", hidden_size // attention_heads)),
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(rope_theta),
            end_token=self._end_token,
        )
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create the Qwen2 model")

        try:
            self._load_safetensors(model_path)
        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _load_safetensors(self, model_path: Path):
        files = sorted(model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors weights found in {model_path}")

        loaded = 0
        for file in files:
            # ACCESS_COPY provides a writable address for ctypes while retaining
            # mmap's copy-on-write, demand-paged behavior. The C++ backend copies
            # each tensor immediately, so the mapping can close after this file.
            with file.open("rb") as stream:
                with mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_COPY) as mapped:
                    header_size = struct.unpack_from("<Q", mapped, 0)[0]
                    header_end = 8 + header_size
                    header = json.loads(mapped[8:header_end].decode("utf-8"))
                    base_address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))

                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        dtype = info["dtype"]
                        if dtype not in _DTYPE_BYTES:
                            raise ValueError(f"Unsupported safetensors dtype {dtype} for {name}")
                        start, end = info["data_offsets"]
                        expected = _DTYPE_BYTES[dtype]
                        for dimension in info["shape"]:
                            expected *= int(dimension)
                        if end - start != expected:
                            raise ValueError(f"Invalid byte size for safetensors tensor {name}")
                        LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                            self._model,
                            name.encode("utf-8"),
                            ctypes.c_void_p(base_address + header_end + start),
                            end - start,
                        )
                        loaded += 1
        if loaded == 0:
            raise ValueError("The safetensors files contained no model weights")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        del top_k, top_p, temperature  # This assignment intentionally uses greedy argmax.
        if not inputs:
            raise ValueError("Qwen2.generate requires at least one input token")
        steps = 128 if max_new_tokens is None else int(max_new_tokens)
        if steps < 0:
            raise ValueError("max_new_tokens must be non-negative")

        output = [int(token) for token in inputs]
        if steps == 0:
            return output

        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)
        prompt = (ctypes.c_int64 * len(output))(*output)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, prompt, len(output))
        )
        for step in range(steps):
            output.append(next_token)
            if next_token == self._end_token or step + 1 == steps:
                break
            token = ctypes.c_int64(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, ctypes.byref(token), 1
                )
            )
        return output
