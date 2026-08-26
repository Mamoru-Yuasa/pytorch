from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from scikit_build_core import build as _backend  # pyrefly: ignore[missing-import]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_HIPIFY_OUTPUT = _REPO_ROOT / "aten/src/ATen/hip/HIPConfig.h.in"
_FALSE_VALUES = {"0", "FALSE", "NO", "OFF", "N"}


def _rocm_build_requested() -> bool:
    use_rocm = os.environ.get("USE_ROCM")
    if use_rocm is not None:
        return use_rocm.upper() not in _FALSE_VALUES
    return sys.platform.startswith("linux") and shutil.which("hipcc") is not None


def _prepare_rocm_source() -> None:
    if not _rocm_build_requested() or _HIPIFY_OUTPUT.exists():
        return
    subprocess.run(
        [sys.executable, "tools/amd_build/build_amd.py"],
        cwd=_REPO_ROOT,
        check=True,
    )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_rocm_source()
    return _backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_rocm_source()
    return _backend.build_editable(wheel_directory, config_settings, metadata_directory)


build_sdist = _backend.build_sdist
get_requires_for_build_editable = _backend.get_requires_for_build_editable
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel
