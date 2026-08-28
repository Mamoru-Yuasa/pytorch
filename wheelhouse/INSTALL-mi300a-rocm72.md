# MI300A ROCm 7.2 PyTorch installation

This guide was prepared with the assistance of a generative AI tool and should
be reviewed before redistribution.

## Requirements

- AMD Instinct MI300A (`gfx942`)
- A compatible AMDGPU/KFD kernel driver on the host
- CPython 3.12 on Linux x86_64 (x86-64-v2 or newer), glibc 2.34 or newer

The wheel bundles the ROCm 7.2 user-space libraries, so a separate ROCm
installation or module is not required at runtime. The kernel driver is not
included and must already be present.

## Installing PyTorch only

Install into a virtual environment so the build does not disturb an existing
PyTorch installation:

```bash
python3.12 -m venv mi300a
source mi300a/bin/activate
python -m pip install ./torch-<version>.mi300a.rocm72-cp312-cp312-linux_x86_64.whl
```

Verify that the APU shared-memory path is active:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("rocm", torch.version.hip)
print("devices", torch.cuda.device_count())

cpu = torch.arange(1024, dtype=torch.float32)
gpu = cpu.to("cuda:0")
if not torch.cuda.apu.is_available(0):
    raise SystemExit("device 0 is not an APU")
if not torch.cuda.apu.is_shared(gpu):
    raise SystemExit("CPU-to-GPU transfer did not use shared memory")
torch.testing.assert_close(gpu.cpu(), cpu)
print("APU shared memory: OK")
PY
```

## Installing the full stack

`install-mi300a-rocm72-stack.sh` additionally installs the TorchVision,
TorchAudio, and Triton versions this build was validated against. Run it from
the directory holding the wheel, inside an activated Python 3.12 environment:

```bash
./install-mi300a-rocm72-stack.sh --yes
```

It replaces any existing `torch`, `torchvision`, `torchaudio`, and `triton`
packages in that environment, then runs an import and GPU-operator check. Pass
the wheel explicitly as an argument if it is not next to the script. The
companion wheels are downloaded from the official PyTorch ROCm 7.2 nightly
index, so the script needs network access.

The validated companion versions are:

- `torchvision==0.30.0.dev20260825+rocm7.2`
- `torchaudio==2.11.0.dev20260825+rocm7.2`
- `triton-rocm==3.8.0+git3f6e4113`

## Known dependency-metadata mismatch

The TorchVision wheel declares a dependency on `torch==2.15.0.dev20260824`,
while this build identifies itself as `2.15.0a0+gitb316297.mi300a.rocm72`. The
native code comes from the same PyTorch source generation, but the version
strings do not satisfy pip's equality check.

The installer therefore uses `--no-deps` for TorchVision and TorchAudio.
Without it, pip would replace this wheel with the stock nightly PyTorch build
and the APU shared-memory support would be lost.

As a consequence, `pip check` reports the TorchVision requirement as unmet even
though imports and native operators work. Other dependency errors are not
expected.

## Validation

This stack was validated on a single MI300A APU. TorchVision GPU NMS,
TorchAudio import, and a Triton-backed `torch.compile` operation all passed.
