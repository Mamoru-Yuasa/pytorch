r"""Automatic shared-memory transfers for supported ROCm APUs."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.cuda._utils import _get_device_index
from torch.types import Device


__all__ = ["SharedTensor", "cpu_view", "is_available", "is_shared", "shared_empty"]


class SharedTensor(NamedTuple):
    r"""CPU and GPU tensor views backed by one unified-memory allocation."""

    gpu: torch.Tensor
    cpu: torch.Tensor

    def synchronize(self) -> None:
        r"""Wait for GPU work before accessing the retained CPU view."""
        torch.cuda.synchronize(self.gpu.device)


def is_available(device: Device = None) -> bool:
    r"""Return whether automatic zero-copy transfers are supported."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    device_index = _get_device_index(device, optional=True)
    return torch._C._cuda_isAPUSupported(device_index)


def is_shared(tensor: torch.Tensor) -> bool:
    r"""Return whether ``tensor`` was produced by an automatic APU alias."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a Tensor, but got {type(tensor).__name__}")
    if torch.version.hip is None:
        return False
    return torch._C._cuda_isUnifiedMemoryAlias(tensor)


def cpu_view(tensor: torch.Tensor, *, synchronize: bool = True) -> torch.Tensor:
    r"""Return a CPU tensor that aliases a supported ROCm APU tensor.

    The returned tensor retains the GPU allocation. By default, all streams on
    the tensor's device are synchronized before the CPU view is returned. If
    ``synchronize`` is ``False``, the caller must ensure that GPU work touching
    the allocation has completed before accessing the CPU view.

    The aliases share an autograd version counter. Synchronization is still
    required when CPU and GPU operations access the allocation in sequence.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a Tensor, but got {type(tensor).__name__}")
    if not tensor.is_cuda:
        raise ValueError("torch.cuda.apu.cpu_view expects a CUDA tensor")
    if not is_available(tensor.device):
        raise RuntimeError("torch.cuda.apu requires a supported ROCm APU")
    cpu = tensor.to(device="cpu", non_blocking=not synchronize)
    if not is_shared(cpu):
        raise RuntimeError("the tensor allocation cannot be shared with the CPU")
    return cpu


def shared_empty(
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    device: Device = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> SharedTensor:
    r"""Allocate an APU tensor with zero-copy CPU and GPU views.

    The allocation comes from PyTorch's normal GPU caching allocator. On
    a supported APU, the CPU and GPU share the physical memory backing this
    allocation, so constructing the CPU view does not allocate or transfer
    tensor data.
    """
    if not is_available(device):
        raise RuntimeError("torch.cuda.apu requires a supported ROCm APU")
    device_index = _get_device_index(device, optional=True)
    gpu = torch.empty(
        tuple(size),
        dtype=dtype,
        device=torch.device("cuda", device_index),
        requires_grad=False,
        memory_format=memory_format,
    )
    cpu = cpu_view(gpu)
    gpu.requires_grad_(requires_grad)
    return SharedTensor(gpu=gpu, cpu=cpu)
