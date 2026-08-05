"""Reports the devices available to torch.

Used to pick where a run happens, to fail fast on `--require-gpu`, and to name
the device in a training notification.
"""
from typing import List

import torch


CPU_DEVICE = 'cpu'
CUDA_DEVICE = 'cuda'


def get_gpu_device_names() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]


def get_device_info() -> dict:
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_device_names': get_gpu_device_names()
    }


def get_gpu_devices(device_info: dict) -> List[str]:
    return device_info.get('gpu_device_names') or []


def get_device_summary(device_info: dict) -> str:
    gpu_devices = get_gpu_devices(device_info)
    if not gpu_devices:
        device_part = 'CPU only'
    else:
        device_part = 'GPU x%d (%s)' % (len(gpu_devices), ', '.join(gpu_devices))
    return '%s [torch: %s]' % (device_part, device_info.get('torch_version'))


def get_default_device() -> str:
    """Returns the device a run should use unless it was asked for one.

    A CPU-only install is the default install, but a run on a machine that has
    a GPU should use it without being told to.
    """
    return CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE
