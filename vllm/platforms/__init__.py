from typing import Optional

import torch

from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Optional[Platform]

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
elif is_hpu:
    from .hpu import HpuPlatform
    current_platform = HpuPlatform()
else:
    current_platform = UnspecifiedPlatform()
__all__ = ['Platform', 'PlatformEnum', 'current_platform']
