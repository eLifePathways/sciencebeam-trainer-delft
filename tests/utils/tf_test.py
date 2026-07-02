from dataclasses import dataclass
from typing import Optional

from sciencebeam_trainer_delft.utils.tf import (
    get_tf_gpu_devices,
    get_tf_device_summary
)


@dataclass
class FakeDevice:
    name: str
    device_type: str
    physical_device_desc: Optional[str] = None


CPU_DEVICE_1 = FakeDevice(name='/device:CPU:0', device_type='CPU')

GPU_DEVICE_1 = FakeDevice(
    name='/device:GPU:0',
    device_type='GPU',
    physical_device_desc='device: 0, name: Tesla T4, pci bus id: 0000:00:04.0'
)

GPU_DEVICE_2 = FakeDevice(
    name='/device:GPU:1',
    device_type='GPU',
    physical_device_desc='device: 1, name: Tesla T4, pci bus id: 0000:00:05.0'
)


class TestGetTfGpuDevices:
    def test_should_return_empty_list_without_gpu_devices(self):
        tf_info = {'tf_version': '2.17.1', 'tf_device_lib': [CPU_DEVICE_1]}
        assert get_tf_gpu_devices(tf_info) == []

    def test_should_return_gpu_devices_only(self):
        tf_info = {
            'tf_version': '2.17.1',
            'tf_device_lib': [CPU_DEVICE_1, GPU_DEVICE_1]
        }
        assert get_tf_gpu_devices(tf_info) == [GPU_DEVICE_1]

    def test_should_return_empty_list_without_device_lib(self):
        tf_info = {'tf_version': '2.17.1', 'tf_device_lib': None}
        assert get_tf_gpu_devices(tf_info) == []


class TestGetTfDeviceSummary:
    def test_should_report_cpu_only(self):
        tf_info = {'tf_version': '2.17.1', 'tf_device_lib': [CPU_DEVICE_1]}
        assert get_tf_device_summary(tf_info) == 'CPU only [tf: 2.17.1]'

    def test_should_report_single_gpu_with_name(self):
        tf_info = {
            'tf_version': '2.17.1',
            'tf_device_lib': [CPU_DEVICE_1, GPU_DEVICE_1]
        }
        assert get_tf_device_summary(tf_info) == 'GPU x1 (Tesla T4) [tf: 2.17.1]'

    def test_should_report_multiple_gpus(self):
        tf_info = {
            'tf_version': '2.17.1',
            'tf_device_lib': [CPU_DEVICE_1, GPU_DEVICE_1, GPU_DEVICE_2]
        }
        assert get_tf_device_summary(tf_info) == (
            'GPU x2 (Tesla T4, Tesla T4) [tf: 2.17.1]'
        )
