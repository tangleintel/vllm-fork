from typing import Optional

import torch

from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Optional[Platform]

is_tpu = False
try:
    # While it's technically possible to install libtpu on a non-TPU machine,
    # this is a very uncommon scenario. Therefore, we assume that libtpu is
    # installed if and only if the machine has TPUs.
    import libtpu  # noqa: F401
    is_tpu = True
except Exception:
    pass

is_hpu = False
try:
    import os
    from importlib import util
    is_hpu = util.find_spec('habana_frameworks') is not None

except Exception:
    pass

if torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif torch.version.hip is not None:
    from .rocm import RocmPlatform
    current_platform = RocmPlatform()
elif is_tpu:
    from .tpu import TpuPlatform
    current_platform = TpuPlatform()
elif is_hpu:
    from .hpu import HpuPlatform
    current_platform = HpuPlatform()
else:
    current_platform = UnspecifiedPlatform()
__all__ = ['Platform', 'PlatformEnum', 'current_platform']
