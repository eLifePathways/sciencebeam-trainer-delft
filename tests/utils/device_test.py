from unittest.mock import patch

from sciencebeam_trainer_delft.utils import device as device_module
from sciencebeam_trainer_delft.utils.device import (
    CPU_DEVICE,
    CUDA_DEVICE,
    get_default_device,
    get_device_info,
    get_device_summary,
    get_gpu_device_names,
    get_gpu_devices
)


TORCH_VERSION = '2.11.0'

GPU_NAME_1 = 'Tesla T4'
GPU_NAME_2 = 'NVIDIA L4'


class TestGetGpuDevices:
    def test_should_return_empty_list_without_gpu_devices(self):
        assert get_gpu_devices({'gpu_device_names': []}) == []

    def test_should_return_empty_list_without_the_key(self):
        assert get_gpu_devices({'torch_version': TORCH_VERSION}) == []

    def test_should_return_the_gpu_device_names(self):
        assert get_gpu_devices({'gpu_device_names': [GPU_NAME_1]}) == [GPU_NAME_1]


class TestGetDeviceSummary:
    def test_should_report_cpu_only(self):
        device_info = {'torch_version': TORCH_VERSION, 'gpu_device_names': []}
        assert get_device_summary(device_info) == f'CPU only [torch: {TORCH_VERSION}]'

    def test_should_report_single_gpu_with_name(self):
        device_info = {
            'torch_version': TORCH_VERSION, 'gpu_device_names': [GPU_NAME_1]
        }
        assert get_device_summary(device_info) == (
            f'GPU x1 ({GPU_NAME_1}) [torch: {TORCH_VERSION}]'
        )

    def test_should_report_multiple_gpus(self):
        device_info = {
            'torch_version': TORCH_VERSION,
            'gpu_device_names': [GPU_NAME_1, GPU_NAME_2]
        }
        assert get_device_summary(device_info) == (
            f'GPU x2 ({GPU_NAME_1}, {GPU_NAME_2}) [torch: {TORCH_VERSION}]'
        )


class TestGetGpuDeviceNames:
    def test_should_return_empty_list_without_cuda(self):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=False):
            assert get_gpu_device_names() == []

    def test_should_name_every_cuda_device(self):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True), \
                patch.object(device_module.torch.cuda, 'device_count', return_value=2), \
                patch.object(
                    device_module.torch.cuda, 'get_device_name',
                    side_effect=[GPU_NAME_1, GPU_NAME_2]
                ):
            assert get_gpu_device_names() == [GPU_NAME_1, GPU_NAME_2]


class TestGetDefaultDevice:
    def test_should_use_the_cpu_without_cuda(self):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=False):
            assert get_default_device() == CPU_DEVICE

    def test_should_use_cuda_when_available(self):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True):
            assert get_default_device() == CUDA_DEVICE


class TestGetDeviceInfo:
    def test_should_report_the_torch_version(self):
        assert get_device_info()['torch_version']

    def test_should_report_gpu_device_names(self):
        assert isinstance(get_device_info()['gpu_device_names'], list)
