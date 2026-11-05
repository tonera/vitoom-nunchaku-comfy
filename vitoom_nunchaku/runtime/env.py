"""Import nunchaku lazily and check its version against the torch ABI."""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError, version

import torch

from vitoom_nunchaku import MIN_NUNCHAKU

log = logging.getLogger("vitoom_nunchaku")

_READY = False
_RUNTIME_VERSION = ""
_LOCAL_VERSION_RE = re.compile(r"cu(\d+(?:\.\d+)?)torch(\d+\.\d+)")
_INSTALL_HINT = (
    "Run this in the Python environment ComfyUI uses:\n"
    "  python vitoom-nunchaku-comfy/scripts/install_wheel.py\n"
    "If Hugging Face is slow, add --mirror, or pass --wheel with a local whl."
)


class NunchakuUnavailable(RuntimeError):
    """The nunchaku wheel is missing, too old, or does not match the current torch ABI."""


def _snapshot_logging():
    root = logging.getLogger()
    return root.level, list(root.handlers), list(root.filters)


def _restore_logging(snapshot) -> None:
    root = logging.getLogger()
    level, handlers, filters = snapshot
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
    root.filters[:] = filters


def _version_tuple(text: str) -> tuple[int, ...]:
    base = text.split("+", 1)[0]
    parts = []
    for piece in base.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cuda_pair(text: str) -> tuple[int, int]:
    parts = text.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def _torch_major_minor(text: str) -> tuple[int, int]:
    parsed = _version_tuple(text)
    if len(parsed) < 2:
        raise NunchakuUnavailable(f"Cannot parse torch version {text!r}.")
    return parsed[0], parsed[1]


def _check_abi(dist_version: str) -> None:
    match = _LOCAL_VERSION_RE.search(dist_version)
    if match is None:
        raise NunchakuUnavailable(
            f"Installed nunchaku version {dist_version!r} has no cu/torch local version. "
            "That usually means the wheel was not published as a pair with the current torch.\n" + _INSTALL_HINT
        )
    wheel_cuda = _cuda_pair(match.group(1))
    wheel_torch = tuple(int(part) for part in match.group(2).split("."))
    runtime_cuda = torch.version.cuda
    if not runtime_cuda:
        raise NunchakuUnavailable("The current torch build has no CUDA version. This plugin does not support CPU or MPS.\n" + _INSTALL_HINT)
    runtime_torch = _torch_major_minor(torch.__version__)
    if wheel_cuda != _cuda_pair(runtime_cuda) or wheel_torch != runtime_torch:
        raise NunchakuUnavailable(
            "The nunchaku wheel does not match the current torch ABI:\n"
            f"  wheel: CUDA {match.group(1)}, torch {match.group(2)} ({dist_version})\n"
            f"  current: CUDA {runtime_cuda}, torch {torch.__version__}\n"
            + _INSTALL_HINT
        )


def ensure_nunchaku():
    """Import nunchaku. Repeated calls do not rewrite the logging configuration again."""
    global _READY, _RUNTIME_VERSION
    if _READY:
        import nunchaku

        return nunchaku

    snapshot = _snapshot_logging()
    try:
        import nunchaku
        import nunchaku._C  # noqa: F401  ABI is exposed only when the extension is imported
        import nunchaku.models.transformers.utils  # noqa: F401  this module calls basicConfig
        import nunchaku.utils  # noqa: F401
    except Exception as exc:
        raise NunchakuUnavailable(
            f"Cannot import nunchaku ({type(exc).__name__}: {exc}).\n" + _INSTALL_HINT
        ) from exc
    finally:
        _restore_logging(snapshot)

    try:
        dist_version = version("nunchaku")
    except PackageNotFoundError as exc:
        raise NunchakuUnavailable(
            "nunchaku can be imported, but its installed distribution metadata was not found.\n" + _INSTALL_HINT
        ) from exc

    try:
        from nunchaku.__version__ import __version__ as runtime_version
    except Exception:
        runtime_version = getattr(nunchaku, "__version__", None) or dist_version.split("+", 1)[0]
    if _version_tuple(runtime_version) < _version_tuple(MIN_NUNCHAKU):
        raise NunchakuUnavailable(
            f"nunchaku >= {MIN_NUNCHAKU} is required, current version is {runtime_version}.\n" + _INSTALL_HINT
        )
    _check_abi(dist_version)
    _READY = True
    _RUNTIME_VERSION = runtime_version
    log.info("nunchaku %s (%s) is ready", runtime_version, dist_version)
    return nunchaku


def require_nunchaku(minimum: str) -> None:
    """Reject a checkpoint whose ArchSpec asks for a newer wheel than the one installed."""
    ensure_nunchaku()
    if _version_tuple(_RUNTIME_VERSION) < _version_tuple(minimum):
        raise NunchakuUnavailable(
            f"This checkpoint needs nunchaku >= {minimum}, current version is {_RUNTIME_VERSION}.\n" + _INSTALL_HINT
        )


def warn_comfyui_version() -> None:
    try:
        import comfyui_version
    except Exception:
        return
    current = getattr(comfyui_version, "__version__", "")
    if current and not current.startswith("0.44"):
        log.warning(
            "Vitoom Nunchaku was verified on ComfyUI %s, current version is %s. A renamed module will fail at load with a missing attribute.",
            "0.44.0",
            current,
        )


def warn_upstream_plugin() -> None:
    try:
        import folder_paths
    except Exception:
        return
    from pathlib import Path

    base = getattr(folder_paths, "base_path", None)
    if not base:
        return
    sibling = Path(base) / "custom_nodes" / "ComfyUI-nunchaku"
    if sibling.exists():
        log.warning(
            "Upstream ComfyUI-nunchaku was detected (%s). Both plugins depend on the package named nunchaku, "
            "so only one wheel version can be installed. Node ids use a Vitoom prefix, so both can appear in the UI.",
            sibling,
        )
