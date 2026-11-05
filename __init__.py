import os
import sys

# ComfyUI loads this directory by file path and does not put it on sys.path.
_ROOT = os.path.dirname(os.path.realpath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vitoom_nunchaku.nodes.extension import comfy_entrypoint  # noqa: E402

__all__ = ["comfy_entrypoint"]
