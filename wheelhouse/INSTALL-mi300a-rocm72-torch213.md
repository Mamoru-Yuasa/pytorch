# PyTorch 2.13 MI300A ROCm 7.2 installation

This guide was prepared with assistance from a generative AI tool and should
be reviewed before redistribution.

## Requirements

- AMD Instinct MI300A (`gfx942`)
- A compatible AMDGPU/KFD kernel driver on the host
- CPython 3.12 on Linux x86_64 (x86-64-v2 or newer), glibc 2.34 or newer

The wheel bundles the ROCm 7.2 user-space libraries. A separate ROCm module is
not required at runtime, but the host kernel driver must already be installed.

## Install the complete stable stack

Create or activate a Python 3.12 virtual environment, then run the installer
next to the wheel:

```bash
python3.12 -m venv mi300a-torch213
source mi300a-torch213/bin/activate
./install-mi300a-rocm72-torch213-stack.sh --yes
```

The script replaces any existing Torch packages in that environment, installs
the custom wheel with the matching stable companions, and runs `pip check`,
version checks, and APU/TorchVision GPU checks when a device is visible.

The validated versions are:

- `torch==2.13.0+mi300a.rocm72`
- `torchvision==0.28.0+rocm7.2`
- `torchaudio==2.11.0+rocm7.2`
- `triton-rocm==3.7.1`

The custom local version `2.13.0+mi300a.rocm72` satisfies TorchVision's
`torch==2.13.0` requirement under PEP 440, so this stack does not need the
`--no-deps` workaround used by the older 2.15 development build.

## Equivalent pip command

The same packages can be installed directly:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
  ./torch-2.13.0+mi300a.rocm72-cp312-cp312-linux_x86_64.whl \
  'torchvision==0.28.0+rocm7.2' \
  'torchaudio==2.11.0+rocm7.2' \
  'triton-rocm==3.7.1'
python -m pip check
```

## Verify shared memory

Run this on an MI300A compute node:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("rocm", torch.version.hip)
print("devices", torch.cuda.device_count())

cpu = torch.arange(1024, dtype=torch.float32)
gpu = cpu.to("cuda:0")
if not torch.cuda.apu.is_available(0):
    raise SystemExit("device 0 is not a supported APU")
if not torch.cuda.apu.is_shared(gpu):
    raise SystemExit("CPU-to-GPU conversion did not use shared memory")
torch.testing.assert_close(gpu.cpu(), cpu)
print("APU shared memory: OK")
PY
```

## ResNet50 validation command

From the `pytorch-image-models` checkout, use the release gate exercised for
this wheel:

```bash
python benchmark.py \
  --model resnet50 \
  --bench train \
  --batch-size 256 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --num-warm-iter 20 \
  --num-bench-iter 100 \
  --no-retry
```
