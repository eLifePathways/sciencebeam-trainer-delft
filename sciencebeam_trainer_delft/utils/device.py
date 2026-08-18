"""Reports the devices available to torch.

Used to pick where a run happens, to fail fast on `--require-gpu`, and to name
the device in a training notification.
"""
import logging
from functools import lru_cache
from typing import List, Optional

import torch


LOGGER = logging.getLogger(__name__)


CPU_DEVICE = 'cpu'
CUDA_DEVICE = 'cuda'


def get_gpu_device_names() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]


def get_gpu_device_capabilities() -> List[str]:
    """Returns the compute capability of each GPU, e.g. `sm_75` for a T4."""
    if not torch.cuda.is_available():
        return []
    return [
        'sm_%d%d' % torch.cuda.get_device_capability(index)
        for index in range(torch.cuda.device_count())
    ]


def get_unsupported_gpu_devices(device_info: dict) -> List[str]:
    """Names the GPUs this torch build has no compiled kernels for.

    A wheel is built for a fixed set of architectures, and running on a card
    outside that set fails at the first kernel launch with `no kernel image is
    available for execution on the device`, which does not say why.
    """
    arch_list = device_info.get('torch_arch_list') or []
    if not arch_list:
        return []
    return [
        capability
        for capability in (device_info.get('gpu_device_capabilities') or [])
        if capability not in arch_list
    ]


def get_device_info() -> dict:
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_device_names': get_gpu_device_names(),
        'gpu_device_capabilities': get_gpu_device_capabilities(),
        # what the installed wheel was compiled for
        'torch_arch_list': list(torch.cuda.get_arch_list())
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


def log_device_info(device_info: dict):
    """Logs the device, and warns if the wheel cannot run on it."""
    LOGGER.info('device_info: %s', device_info)
    unsupported = get_unsupported_gpu_devices(device_info)
    if unsupported:
        LOGGER.warning(
            'this torch build has no kernels for %s (built for %s):'
            ' a different CUDA wheel is required',
            ', '.join(unsupported), ', '.join(device_info['torch_arch_list'])
        )


@lru_cache(maxsize=1)
def log_device_info_once(device: Optional[str] = None) -> None:
    """Logs the selected device and the device info, once per process.

    Both the training CLI and `Sequence` call this: a serving consumer has no
    other way to learn which device it got, and a tagging service constructs a
    `Sequence` per model, so the lines must not repeat. Call it with the
    resolved device rather than the requested one, so that both call sites pass
    the same value and only the first one logs.
    """
    if device:
        LOGGER.info('using device: %s', device)
    log_device_info(get_device_info())
    if device and torch.device(device).type == CUDA_DEVICE and not torch.cuda.is_available():
        LOGGER.warning(
            'device %r was requested, but torch reports no CUDA device:'
            ' moving the model to it will fail',
            device
        )


def validate_device(device: str, source: Optional[str] = None) -> str:
    """Returns the device, or fails naming the value torch cannot use.

    Without this an unusable value reaches `.to(device)` much later, where the
    message is about a tensor rather than about the value that was asked for.
    A device torch can parse but has no hardware for is not rejected here:
    `log_device_info_once` warns about it instead, since an image can set the
    value ahead of the hardware it will run on.
    """
    try:
        torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('invalid device: %r%s' % (
            device,
            ' (from %s)' % source if source else ''
        )) from exc
    return device


def get_default_device() -> str:
    """Returns the device a run should use unless it was asked for one.

    A CPU-only install is the default install, but a run on a machine that has
    a GPU should use it without being told to.
    """
    return CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE
