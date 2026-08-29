# torch.cuda

```{eval-rst}
.. automodule:: torch.cuda
```

```{eval-rst}
.. currentmodule:: torch.cuda
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    can_device_access_peer
    check_error
    current_blas_handle
    current_solver_handle
    current_device
    current_stream
    cudart
    default_stream
    device
    device_count
    device_memory_used
    device_of
    get_arch_list
    get_device_capability
    get_device_name
    get_device_properties
    get_gencode_flags
    get_stream_from_external
    get_sync_debug_mode
    init
    ipc_collect
    is_available
    is_bf16_supported
    is_initialized
    is_tf32_supported
    memory_usage
    set_device
    set_stream
    set_sync_debug_mode
    stream
    synchronize
    utilization
    temperature
    power_draw
    clock_rate
    AcceleratorError
    OutOfMemoryError
```

## Random Number Generator

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed

```

## Communication collectives

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    comm.broadcast
    comm.broadcast_coalesced
    comm.reduce_add
    comm.reduce_add_coalesced
    comm.scatter
    comm.gather
```

## Streams and events

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event
```

## Graphs (beta)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables
```

## ROCm APU unified memory

On a supported integrated ROCm device, PyTorch automatically avoids a physical
copy for eligible CPU-to-GPU and GPU-to-CPU `Tensor.to` conversions. Runtime
device properties and pointer capabilities determine whether an allocation can
be shared; device model names, cluster settings, and environment-specific paths
are not used. Unsupported devices and allocations retain the normal copy path.

The APU-enabled branch can be installed directly into an active virtual
environment. With a ROCm toolchain on `PATH`, pip automatically prepares the
ROCm sources before building the wheel:

```bash
python -m pip install \
    "git+https://github.com/Mamoru-Yuasa/pytorch.git@mi300a-apu-shared-memory-v2.13"
```

Automatic sharing applies when `copy=False`, the dtype and layout are
unchanged, the tensor is non-overlapping and dense, and the conversion does not
need an autograd edge. This includes transfers made under `torch.no_grad()`, as
used when moving module parameters. A CPU allocation is registered with HIP for
device access, while a GPU allocation uses its host mapping. The returned
tensor retains the source allocation and shares its autograd version counter.

```python
import torch

cpu = torch.ones(1024, 1024)
gpu = cpu.to("cuda")
result = gpu.square()
result_cpu = result.cpu()

if torch.cuda.apu.is_shared(gpu):
    print("CPU-to-GPU copy was elided")
```

Because `copy=False` permits aliasing, in-place writes through either result are
visible through the other alias. Use `to(..., copy=True)` when independent
storage is required. The source storage cannot be resized while an alias is
alive. GPU-to-CPU conversion waits for the streams associated with the
allocation by default; callers using `non_blocking=True` retain the usual
responsibility for stream synchronization. Autograd-requiring tensors,
forward-mode dual tensors, inference mode, dtype conversions, unsupported
layouts, and unsuccessful runtime registration automatically fall back to a
copy.

`torch.cuda.apu.is_available()` reports the runtime capability and
`torch.cuda.apu.is_shared(tensor)` identifies a conversion whose copy was
elided. `shared_empty` and `cpu_view` remain available when an explicitly paired
allocation is useful. See the
[ROCm unified-memory documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html)
for allocator and host-registration behavior.

```{eval-rst}
.. automodule:: torch.cuda.apu
.. currentmodule:: torch.cuda.apu

.. autosummary::
    :toctree: generated
    :nosignatures:

    is_available
    is_shared
    shared_empty
    cpu_view
    SharedTensor

.. currentmodule:: torch.cuda
```

(cuda-memory-management-api)=

```{eval-rst}
.. automodule:: torch.cuda.memory
```

```{eval-rst}
.. currentmodule:: torch.cuda.memory
```

## Memory management

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     get_per_process_memory_fraction
     list_gpu_processes
     mem_get_info
     memory_stats
     memory_stats_as_nested_dict
     reset_accumulated_memory_stats
     host_memory_stats
     host_memory_stats_as_nested_dict
     reset_accumulated_host_memory_stats
     memory_summary
     memory_snapshot
     memory_allocated
     max_memory_allocated
     reset_max_memory_allocated
     memory_reserved
     max_memory_reserved
     set_per_process_memory_fraction
     memory_cached
     max_memory_cached
     reset_max_memory_cached
     reset_peak_memory_stats
     reset_peak_host_memory_stats
     caching_allocator_alloc
     caching_allocator_delete
     get_allocator_backend
     CUDAPluggableAllocator
     change_current_allocator
     MemPool
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    caching_allocator_disabled
    caching_allocator_enable
```

```{eval-rst}
.. currentmodule:: torch.cuda
```

```{eval-rst}
.. autoclass:: torch.cuda.use_mem_pool
```

```{eval-rst}
.. currentmodule:: torch.cuda.nccl
```

```{eval-rst}
.. autofunction:: version
```

```{eval-rst}
.. currentmodule:: torch.cuda.profiler

.. autosummary::
    :toctree: generated
    :nosignatures:

    profile
    start
    stop
```

```{eval-rst}
.. currentmodule:: torch.cuda
```

## NVIDIA Tools Extension (NVTX)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    nvtx.mark
    nvtx.range_push
    nvtx.range_pop
    nvtx.range
    nvtx.range_end
    nvtx.range_start
```

## Jiterator (beta)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    jiterator._create_jit_fn
    jiterator._create_multi_output_jit_fn
```

## TunableOp

Some operations could be implemented using more than one library or more than
one technique. For example, a GEMM could be implemented for CUDA or ROCm using
either the cublas/cublasLt libraries or hipblas/hipblasLt libraries,
respectively. How does one know which implementation is the fastest and should
be chosen? That's what TunableOp provides. Certain operators have been
implemented using multiple strategies as Tunable Operators. At runtime, all
strategies are profiled and the fastest is selected for all subsequent
operations.

See the {doc}`documentation <cuda.tunable>` for information on how to use it.

```{toctree}
:hidden: true

cuda.tunable
```

## Stream Sanitizer (prototype)

CUDA Sanitizer is a prototype tool for detecting synchronization errors between streams in PyTorch.
See the {doc}`documentation <cuda._sanitizer>` for information on how to use it.

```{toctree}
:hidden: true

cuda._sanitizer
```

## GPUDirect Storage (prototype)

The APIs in `torch.cuda.gds` provide thin wrappers around certain cuFile APIs that allow
direct memory access transfers between GPU memory and storage, avoiding a bounce buffer in the CPU. See the
[cufile api documentation](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api)
for more details.

These APIs can be used in versions greater than or equal to CUDA 12.6. In order to use these APIs, one must
ensure that their system is appropriately configured to use GPUDirect Storage per the
[GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/contents.html).

See the docs for {class}`~torch.cuda.gds.GdsFile` for an example of how to use these.

```{eval-rst}
.. currentmodule:: torch.cuda.gds
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    gds_register_buffer
    gds_deregister_buffer
    GdsFile

```

## Green Contexts (experimental)

`torch.cuda.green_contexts` provides thin wrappers around the CUDA Green Context APIs
to enable more general carveout of SM resources for CUDA kernels.

These APIs can be used in PyTorch with CUDA versions greater than or equal to 12.8.

See the docs for {class}`~torch.cuda.green_contexts.GreenContext` for an example of how to use these.

```{eval-rst}
.. currentmodule:: torch.cuda.green_contexts
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    GreenContext
```


% This module needs to be documented. Adding here in the meantime

% for tracking purposes

```{eval-rst}
.. py:module:: torch.cuda.comm
```

```{eval-rst}
.. py:module:: torch.cuda.gds
```

```{eval-rst}
.. py:module:: torch.cuda.green_contexts
```

```{eval-rst}
.. py:module:: torch.cuda.jiterator
```

```{eval-rst}
.. py:module:: torch.cuda.nccl

.. autofunction:: torch.cuda.nccl.is_available
```

```{eval-rst}
.. automodule:: torch.cuda.nvtx
```

```{eval-rst}
.. currentmodule:: torch.cuda.nvtx
```

```{eval-rst}
.. py:module:: torch.cuda.profiler
```

```{eval-rst}
.. py:module:: torch.cuda.sparse
```

```{eval-rst}
.. toctree::
    :hidden:

    cuda.aliases.md
```
