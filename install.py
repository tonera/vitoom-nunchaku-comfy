#!/usr/bin/env python3
"""ComfyUI Manager runs this after the node is installed from the UI.

Manager calls this file with ComfyUI's own Python. The script selects the nunchaku wheel that matches the current Python, torch, and CUDA, then installs it.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "scripts" / "install_wheel.py"


if __name__ == "__main__":
    sys.argv = [str(_SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(_SCRIPT), run_name="__main__")
