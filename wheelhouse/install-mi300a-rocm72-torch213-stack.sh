#!/usr/bin/env bash
# Install the PyTorch 2.13 MI300A ROCm 7.2 wheel and its stable companion
# packages. This script was prepared with assistance from a generative AI tool
# and should be reviewed before use.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-mi300a-rocm72-torch213-stack.sh --yes [TORCH_WHEEL] [--allow-system]

Replaces torch, torchvision, torchaudio, triton, and triton-rocm in the active
CPython 3.12 environment with the validated PyTorch 2.13 ROCm 7.2 stack.

  --yes            Required. Confirms that the packages above may be replaced.
  TORCH_WHEEL      Path or URL to the custom torch wheel. Defaults to the
                   PyTorch 2.13 MI300A wheel next to this script.
  --allow-system   Permit installation into a non-virtual environment.

Companion wheels are downloaded from the official PyTorch ROCm 7.2 index.
USAGE
  exit 2
}

confirmed=
allow_system=
torch_wheel=
for arg in "$@"; do
  case ${arg} in
    --yes) confirmed=1 ;;
    --allow-system) allow_system=1 ;;
    -h|--help) usage ;;
    -*)
      echo "unknown option: ${arg}" >&2
      usage
      ;;
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
  torch_wheel=${script_dir}/torch-2.13.0+mi300a.rocm72-cp312-cp312-linux_x86_64.whl
fi
if [[ ${torch_wheel} != http://* && ${torch_wheel} != https://* && ! -f ${torch_wheel} ]]; then
  echo "torch wheel does not exist: ${torch_wheel}" >&2
  exit 2
fi

stable_index=https://download.pytorch.org/whl/rocm7.2
torchvision_version=0.28.0+rocm7.2
torchaudio_version=2.11.0+rocm7.2
triton_version=3.7.1

python - "${allow_system:-}" <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"error: CPython 3.12 is required, got {sys.version.split()[0]}")

allow_system = bool(sys.argv[1])
in_venv = sys.prefix != sys.base_prefix
if not in_venv and not allow_system:
    raise SystemExit(
        "error: refusing to modify a non-virtual environment.\n"
        "Create and activate a virtual environment (venv or conda), or pass "
        "--allow-system to override."
    )
PY

python -m pip uninstall --yes torch torchvision torchaudio triton triton-rocm
python -m pip install \
  --extra-index-url "${stable_index}" \
  "${torch_wheel}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torchaudio_version}" \
  "triton-rocm==${triton_version}"
python -m pip check

python - <<'PY'
import torch
import torchaudio
import torchvision
import triton
from torchvision.ops import nms

expected = {
    "torch": "2.13.0+mi300a.rocm72",
    "torchvision": "0.28.0+rocm7.2",
    "torchaudio": "2.11.0+rocm7.2",
    "triton": "3.7.1",
}
actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "triton": triton.__version__,
}
if actual != expected:
    raise SystemExit(f"error: unexpected package versions: {actual}")
if not torchvision.extension._has_ops():
    raise SystemExit("error: torchvision C++ operators are unavailable")

print("versions", actual)
print("rocm", torch.version.hip)
print("devices", torch.cuda.device_count())
if torch.cuda.device_count():
    if not torch.cuda.apu.is_available(0):
        raise SystemExit("error: CUDA device 0 is not a supported ROCm APU")
    cpu = torch.arange(1024, dtype=torch.float32)
    gpu = cpu.to("cuda:0")
    if not torch.cuda.apu.is_shared(gpu):
        raise SystemExit("error: CPU-to-GPU conversion did not use shared memory")
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5]], device="cuda"
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    if not torch.equal(nms(boxes, scores, 0.3).cpu(), torch.tensor([0])):
        raise SystemExit("error: TorchVision GPU NMS returned an unexpected result")
    print("apu_shared_memory", "ok")
    print("torchvision_gpu_nms", "ok")
else:
    print("GPU checks skipped: no CUDA/ROCm device is visible")
PY

echo "PyTorch 2.13 MI300A ROCm 7.2 stack installed successfully."
