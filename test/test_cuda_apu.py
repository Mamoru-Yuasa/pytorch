# Owner(s): ["module: cuda"]

import gc
import mmap

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestAPUAPI(TestCase):
    def test_rejects_cpu_tensor(self):
        with self.assertRaisesRegex(ValueError, "expects a CUDA tensor"):
            torch.cuda.apu.cpu_view(torch.empty(1))

    def test_cpu_tensor_is_not_shared(self):
        self.assertFalse(torch.cuda.apu.is_shared(torch.empty(1)))

    def test_tensor_without_storage_is_not_shared(self):
        sparse = torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=torch.int64), torch.empty(0), (1,)
        )
        self.assertFalse(torch.cuda.apu.is_shared(sparse))


class TestAPUDevice(TestCase):
    def _require_apu(self, device):
        if not torch.cuda.apu.is_available(device):
            self.skipTest("requires a ROCm APU with unified memory")

    @parametrize("dtype", [torch.float32, torch.int64])
    def test_standard_to_aliases_cpu_memory(self, device, dtype):
        self._require_apu(device)
        cpu = torch.arange(32, dtype=dtype)
        allocated = torch.cuda.memory_allocated(device)

        gpu = cpu.to(device)

        self.assertTrue(torch.cuda.apu.is_shared(gpu))
        self.assertEqual(torch.cuda.memory_allocated(device), allocated)
        self.assertEqual(gpu, cpu)
        gpu.add_(1)
        torch.cuda.synchronize(device)
        self.assertEqual(cpu, torch.arange(32, dtype=dtype) + 1)

    def test_standard_cpu_aliases_gpu_memory(self, device):
        self._require_apu(device)
        gpu = torch.arange(32, dtype=torch.float32, device=device)
        allocated = torch.cuda.memory_allocated(device)

        cpu = gpu.cpu()

        self.assertTrue(torch.cuda.apu.is_shared(cpu))
        self.assertEqual(torch.cuda.memory_allocated(device), allocated)
        cpu.add_(1)
        self.assertEqual(gpu, torch.arange(32, dtype=torch.float32, device=device) + 1)

    @parametrize("direction", ["cpu_to_gpu", "gpu_to_cpu"])
    def test_copy_true_preserves_independent_storage(self, device, direction):
        self._require_apu(device)
        if direction == "cpu_to_gpu":
            source = torch.arange(32, dtype=torch.float32)
            copied = source.to(device, copy=True)
        else:
            source = torch.arange(32, dtype=torch.float32, device=device)
            copied = source.to("cpu", copy=True)

        self.assertFalse(torch.cuda.apu.is_shared(copied))
        copied.add_(1)
        self.assertEqual(
            source, torch.arange(32, dtype=torch.float32, device=source.device)
        )

    def test_conversion_and_autograd_use_copy_fallback(self, device):
        self._require_apu(device)
        cpu = torch.arange(8, dtype=torch.float32, requires_grad=True)
        gpu = cpu.to(device)
        converted = cpu.detach().to(device=device, dtype=torch.float64)

        self.assertFalse(torch.cuda.apu.is_shared(gpu))
        self.assertFalse(torch.cuda.apu.is_shared(converted))
        gpu.sum().backward()
        self.assertEqual(cpu.grad, torch.ones_like(cpu))

    def test_no_grad_parameter_transfer_uses_alias(self, device):
        self._require_apu(device)
        cpu = torch.arange(8, dtype=torch.float32, requires_grad=True)

        with torch.no_grad():
            gpu = cpu.to(device)

        self.assertTrue(torch.cuda.apu.is_shared(gpu))
        self.assertFalse(gpu.requires_grad)

    def test_module_to_uses_alias_with_autograd(self, device):
        self._require_apu(device)
        module = torch.nn.Linear(8, 4)

        module.to(device)

        self.assertTrue(torch.cuda.apu.is_shared(module.weight))
        self.assertTrue(torch.cuda.apu.is_shared(module.bias))
        self.assertTrue(module.weight.requires_grad)
        output = module(torch.ones(2, 8, device=device))
        output.sum().backward()
        self.assertEqual(module.weight.grad.device, torch.device(device))

    def test_forward_ad_uses_copy_fallback(self, device):
        self._require_apu(device)
        primal = torch.arange(8, dtype=torch.float32)
        tangent = torch.ones_like(primal)

        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            gpu = dual.to(device)
            gpu_primal, gpu_tangent = torch.autograd.forward_ad.unpack_dual(gpu)

        self.assertFalse(torch.cuda.apu.is_shared(gpu))
        self.assertEqual(gpu_primal, primal)
        self.assertEqual(gpu_tangent, tangent)

    def test_inference_mode_uses_copy_fallback(self, device):
        self._require_apu(device)
        cpu = torch.arange(8, dtype=torch.float32)

        with torch.inference_mode():
            gpu = cpu.to(device)

        self.assertFalse(torch.cuda.apu.is_shared(gpu))
        self.assertTrue(gpu.is_inference())

        with torch.inference_mode():
            inference_cpu = torch.arange(8, dtype=torch.float32)
        normal_gpu = inference_cpu.to(device)
        self.assertFalse(torch.cuda.apu.is_shared(normal_gpu))
        self.assertFalse(normal_gpu.is_inference())

    def test_alias_preserves_shape_strides_and_lifetime(self, device):
        self._require_apu(device)
        cpu = torch.arange(35, dtype=torch.float32).reshape(5, 7).t()
        gpu = cpu.to(device)

        self.assertEqual(gpu.size(), cpu.size())
        self.assertEqual(gpu.stride(), cpu.stride())
        del cpu
        gc.collect()
        self.assertEqual(gpu, torch.arange(35, dtype=torch.float32).reshape(5, 7).t())

    def test_source_storage_is_resizable_after_alias_dies(self, device):
        self._require_apu(device)
        cpu = torch.arange(8, dtype=torch.float32)
        gpu = cpu.to(device)

        with self.assertRaisesRegex(RuntimeError, "not resizable"):
            cpu.resize_(16)
        del gpu
        gc.collect()
        cpu.resize_(16)
        self.assertEqual(cpu.numel(), 16)

    def test_aliases_sharing_a_host_page_stay_valid(self, device):
        self._require_apu(device)
        page = mmap.PAGESIZE
        held = []
        by_page = {}
        pair = None
        for _ in range(512):
            candidate = torch.arange(8, dtype=torch.float32)
            held.append(candidate)
            start = candidate.data_ptr()
            end = start + candidate.numel() * candidate.element_size() - 1
            if start // page != end // page:
                continue
            other = by_page.setdefault(start // page, candidate)
            if other is not candidate:
                pair = (other, candidate)
                break
        if pair is None:
            self.skipTest("could not place two CPU tensors in one host page")

        first, second = pair
        first_gpu = first.to(device)
        second_gpu = second.to(device)
        self.assertTrue(torch.cuda.apu.is_shared(first_gpu))
        self.assertTrue(torch.cuda.apu.is_shared(second_gpu))

        del first_gpu
        gc.collect()

        second_gpu.add_(1)
        torch.cuda.synchronize(device)
        self.assertEqual(second, torch.arange(8, dtype=torch.float32) + 1)

    def test_alias_spanning_adjacent_host_registrations(self, device):
        self._require_apu(device)
        page = mmap.PAGESIZE
        mapping = mmap.mmap(-1, 2 * page)
        first = torch.frombuffer(mapping, dtype=torch.uint8, count=page)
        crossing = torch.frombuffer(
            mapping, dtype=torch.uint8, count=256, offset=page - 64
        )
        expected = torch.arange(256, dtype=torch.uint8)
        crossing.copy_(expected)

        first_gpu = first.to(device)
        crossing_gpu = crossing.to(device)

        self.assertTrue(torch.cuda.apu.is_shared(first_gpu))
        self.assertTrue(torch.cuda.apu.is_shared(crossing_gpu))
        crossing_gpu.add_(1)
        torch.cuda.synchronize(device)
        self.assertEqual(crossing, expected + 1)

        del first_gpu
        gc.collect()
        crossing_gpu.add_(1)
        torch.cuda.synchronize(device)
        self.assertEqual(crossing, expected + 2)

    def test_record_stream_keeps_registered_storage_alive(self, device):
        self._require_apu(device)
        cpu = torch.zeros(1024 * 1024, dtype=torch.float32)
        gpu = cpu.to(device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            gpu.add_(1)
            gpu.record_stream(stream)
        del gpu
        gc.collect()

        self.assertEqual(cpu, torch.ones_like(cpu))

    @parametrize("dtype", [torch.float32, torch.int64])
    def test_shared_empty_zero_copy(self, device, dtype):
        self._require_apu(device)
        shared = torch.cuda.apu.shared_empty((4, 8), dtype=dtype, device=device)
        expected = torch.arange(32, dtype=dtype).reshape(4, 8)

        shared.cpu.copy_(expected)
        shared.gpu.mul_(2)
        shared.synchronize()

        self.assertTrue(torch.cuda.apu.is_shared(shared.cpu))
        self.assertEqual(shared.cpu, expected * 2)


instantiate_device_type_tests(TestAPUDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
