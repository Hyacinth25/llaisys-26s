import ctypes

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


llaisysQwen2Model_t = ctypes.c_void_p


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelLoadWeight.argtypes = [
        llaisysQwen2Model_t,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2ModelLoadWeight.restype = None

    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64


__all__ = ["LlaisysQwen2Meta", "llaisysQwen2Model_t", "load_qwen2"]
