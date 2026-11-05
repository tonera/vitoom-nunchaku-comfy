#!/usr/bin/env python3
"""Pick a nunchaku wheel for the current Python, torch, and CUDA, then install it.

Install the wheel with ``pip install --no-deps`` first so pip does not change the torch that is already installed,
then install requirements.txt. transformers and huggingface_hub are not upgraded when they already satisfy the dependencies.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

REPO = "tonera/vitoom-nunchaku"
DEFAULT_VERSION = "1.4.1"
_PINNED = (
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "einops",
    "sentencepiece",
)


def _platform_tag() -> str:
    system = sys.platform
    machine = platform.machine().lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux_x86_64"
    if system == "win32" and machine in ("x86_64", "amd64"):
        return "win_amd64"
    raise SystemExit(f"Unsupported platform: {system} {machine}")


def _torch_facts() -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is not installed in this environment. Install a PyTorch build that matches the GPU before installing nunchaku.") from exc
    cuda = torch.version.cuda
    if not cuda:
        raise SystemExit(f"The current torch {torch.__version__} is not a CUDA build.")
    numbers = torch.__version__.split("+", 1)[0].split(".")
    if len(numbers) < 2:
        raise SystemExit(f"Cannot parse torch version {torch.__version__}")
    return f"cu{cuda}", f"torch{numbers[0]}.{numbers[1]}"


def wheel_filename(version: str) -> str:
    cuda, torch_tag = _torch_facts()
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"nunchaku-{version}+{cuda}{torch_tag}-{py}-{py}-{_platform_tag()}.whl"


def _closest(files: list[str], wanted: str) -> list[str]:
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = [name for name in files if name.endswith(".whl") and "nunchaku" in name]
    ranked = []
    for name in wheels:
        score = 0
        if py in name:
            score += 2
        if "+cu" in wanted and wanted.split("+", 1)[1].split("-", 1)[0] in name:
            score += 2
        ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in ranked[:12]]


def _resolve_wheel(version: str, mirror: bool) -> str:
    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    filename = wheel_filename(version)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to look up the wheel. You can also pass --wheel with a local file.") from exc

    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    files = list(api.list_repo_files(REPO))
    if filename not in files:
        options = "\n".join(f"  {name}" for name in _closest(files, filename)) or "  (no whl in the repository)"
        raise SystemExit(
            f"Repository {REPO} does not contain {filename}.\n"
            "Refusing to guess an install. Closest files:\n"
            f"{options}\n"
            "You can also install a local whl with --wheel."
        )
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{base}/{REPO}/resolve/main/{quote(filename)}"


def _constraint_file() -> Path | None:
    lines = []
    for name in _PINNED:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    if not lines:
        return None
    handle = tempfile.NamedTemporaryFile("w", prefix="vitoom-nunchaku-pins-", suffix=".txt", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return Path(handle.name)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the nunchaku wheel that matches the current torch/CUDA")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="nunchaku version, default 1.4.1")
    parser.add_argument("--wheel", help="Path to a local whl. When set, Hugging Face is not contacted")
    parser.add_argument("--mirror", action="store_true", help="Use https://hf-mirror.com")
    parser.add_argument("--skip-requirements", action="store_true", help="Install only the wheel, not diffusers and the other dependencies")
    args = parser.parse_args()

    if args.wheel:
        wheel = str(Path(args.wheel).expanduser().resolve())
        if not Path(wheel).is_file():
            raise SystemExit(f"Wheel not found: {wheel}")
    else:
        wheel = _resolve_wheel(args.version, args.mirror)

    before = {name: metadata.version(name) for name in ("transformers", "huggingface_hub") if _installed(name)}
    _run([sys.executable, "-m", "pip", "install", "--no-deps", wheel])
    if not args.skip_requirements:
        requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
        command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        pins = _constraint_file()
        if pins is not None:
            command.extend(["-c", str(pins)])
        _run(command)
    _run([
        sys.executable,
        "-c",
        "import nunchaku, nunchaku._C; from nunchaku.__version__ import __version__; print(__version__)",
    ])
    for name, old in before.items():
        current = metadata.version(name)
        if current != old:
            raise SystemExit(f"{name} changed from {old} to {current}. Check the dependency pins; this script should not change it.")
    print("nunchaku install finished.")


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    main()
