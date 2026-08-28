#!/usr/bin/env bash
# Install the MI300A ROCm 7.2 PyTorch wheel together with a matching
# TorchVision, TorchAudio, and Triton.
#
# This script was prepared with the assistance of a generative AI tool and
# should be reviewed before use.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-mi300a-rocm72-stack.sh --yes [TORCH_WHEEL] [--allow-system]

Replaces torch, torchvision, torchaudio, and triton in the active Python 3.12
environment with the MI300A ROCm 7.2 build.

  --yes            Required. Confirms that the packages above may be replaced.
  TORCH_WHEEL      Path or URL to the MI300A torch wheel. Defaults to a
                   torch-*.mi300a.rocm72-*.whl file next to this script.
  --allow-system   Permit installing into a non-virtual environment.

The companion wheels are downloaded from the official PyTorch ROCm 7.2 nightly
index, so network access is required unless they are already cached.
USAGE
  exit 2
}

confirmed=
allow_system=
torch_wheel=
for arg in "$@"; do
  case "${arg}" in
    --yes) confirmed=1 ;;
    --allow-system) allow_system=1 ;;
    -h|--help) usage ;;
    -*) echo "unknown option: ${arg}" >&2; usage ;;
    *)
      if [[ -n ${torch_wheel} ]]; then
        echo "more than one wheel given" >&2
        usage
      fi
      torch_wheel=${arg}
      ;;
  esac
done
[[ -n ${confirmed} ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${torch_wheel} ]]; then
  torch_wheel=$(find "${script_dir}" -maxdepth 1 -type f \
    -name 'torch-*.mi300a.rocm72-*.whl' -print -quit)
  if [[ -z ${torch_wheel} ]]; then
    echo "no torch-*.mi300a.rocm72-*.whl next to this script; pass one explicitly" >&2
    usage
  fi
fi
# A URL is handed to pip untouched; only a local path is checked here.
if [[ ${torch_wheel} != http://* && ${torch_wheel} != https://* && ! -f ${torch_wheel} ]]; then
  echo "torch wheel does not exist: ${torch_wheel}" >&2
  exit 2
fi

# These are the versions this stack was validated against. The MI300A wheel
# reports a custom version string, so TorchVision and TorchAudio are installed
# with --no-deps to stop pip from pulling the stock nightly torch over it.
nightly_index=https://download.pytorch.org/whl/nightly/rocm7.2
torchvision_version=0.30.0.dev20260825+rocm7.2
torchaudio_version=2.11.0.dev20260825+rocm7.2
triton_version=3.8.0+git3f6e4113

python - "${allow_system:-}" <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"error: CPython 3.12 is required, got {sys.version.split()[0]}")

allow_system = bool(sys.argv[1])
in_venv = sys.prefix != sys.base_prefix
if not in_venv and not allow_system:
    raise SystemExit(
        "error: refusing to modify a non-virtual environment.\n"
        "Create and activate a virtual environment (python -m venv, conda create, ...)\n"
        "or pass --allow-system to override."
    )
PY

python -m pip uninstall --yes torch torchvision torchaudio triton triton-rocm
python -m pip install "${torch_wheel}"
python -m pip install "numpy" "pillow!=8.3.*,>=5.3.0"
python -m pip install \
  --index-url "${nightly_index}" \
  "triton-rocm==${triton_version}"
python -m pip install \
  --no-deps \
  --index-url "${nightly_index}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torchaudio_version}"

python - <<'PY'
import torch
import torchaudio
import torchvision
import triton
from torchvision.ops import nms

if ".mi300a." not in torch.__version__:
    raise SystemExit(f"error: expected an MI300A torch build, got {torch.__version__}")
if not torchvision.extension._has_ops():
    raise SystemExit("error: torchvision C++ operators are unavailable")

boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5]])
scores = torch.tensor([0.9, 0.8])
print("nms", nms(boxes, scores, 0.5))
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("triton", triton.__version__)
print("rocm", torch.version.hip)
print("devices", torch.cuda.device_count())
PY

echo "MI300A PyTorch companion stack installed successfully."
