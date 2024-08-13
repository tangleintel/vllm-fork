import torch

from vllm.utils import is_hpu

from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Platform

try:
    import libtpu
except ImportError:
    libtpu = None

if libtpu is not None:
    # people might install pytorch built with cuda but run on tpu
    # so we need to check tpu first
    from .tpu import TpuPlatform
    current_platform = TpuPlatform()
elif torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif torch.version.hip is not None:
    from .rocm import RocmPlatform
    current_platform = RocmPlatform()
elif is_hpu():
    from .hpu import HpuPlatform
    current_platform = HpuPlatform()
else:
    current_platform = UnspecifiedPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform']
