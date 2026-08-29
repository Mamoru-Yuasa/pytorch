import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools import build_backend

from torch.testing._internal.common_utils import run_tests, TestCase


class TestBuildBackend(TestCase):
    @mock.patch.dict(os.environ, {"USE_ROCM": "0"})
    @mock.patch("tools.build_backend.shutil.which", return_value="/opt/rocm/bin/hipcc")
    def test_rocm_build_can_be_disabled(self, _which):
        self.assertFalse(build_backend._rocm_build_requested())

    @mock.patch.dict(os.environ, {"USE_ROCM": "1"})
    @mock.patch("tools.build_backend.subprocess.run")
    def test_prepare_rocm_source(self, run):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aten/src/ATen/hip/HIPConfig.h.in"
            with (
                mock.patch.object(build_backend, "_REPO_ROOT", root),
                mock.patch.object(build_backend, "_HIPIFY_OUTPUT", output),
            ):
                build_backend._prepare_rocm_source()

        run.assert_called_once_with(
            [sys.executable, "tools/amd_build/build_amd.py"],
            cwd=root,
            check=True,
        )

    @mock.patch.dict(os.environ, {"USE_ROCM": "1"})
    @mock.patch("tools.build_backend.subprocess.run")
    def test_prepared_rocm_source_is_not_modified_again(self, run):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aten/src/ATen/hip/HIPConfig.h.in"
            output.parent.mkdir(parents=True)
            output.touch()
            with mock.patch.object(build_backend, "_HIPIFY_OUTPUT", output):
                build_backend._prepare_rocm_source()

        run.assert_not_called()


if __name__ == "__main__":
    run_tests()
