from vitoom_nunchaku.archs.base import get_spec, registered_model_classes

# Importing registers the specs. This does not import nunchaku.
from vitoom_nunchaku.archs import anima as _anima  # noqa: F401
from vitoom_nunchaku.archs import flux1 as _flux1  # noqa: F401
from vitoom_nunchaku.archs import flux2 as _flux2  # noqa: F401
from vitoom_nunchaku.archs import flux3 as _flux3  # noqa: F401
from vitoom_nunchaku.archs import qwen_image21 as _qwen_image21  # noqa: F401
from vitoom_nunchaku.archs import sdxl as _sdxl  # noqa: F401
from vitoom_nunchaku.archs import zimage as _zimage  # noqa: F401

__all__ = ["get_spec", "registered_model_classes"]
