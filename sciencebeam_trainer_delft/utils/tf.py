import re
from typing import Any, List


try:
    from tensorflow import __version__ as tf_version
    from tensorflow.python.client import device_lib as tf_device_lib
except ImportError:
    tf_version = None
    tf_device_lib = None


GPU_DEVICE_TYPE = 'GPU'

GPU_PHYSICAL_DEVICE_DESC_NAME_PATTERN = re.compile(r'name:\s*([^,]+)')


def get_tf_info() -> dict:
    return {
        'tf_version': tf_version,
        'tf_device_lib': tf_device_lib.list_local_devices() if tf_device_lib else None
    }


def get_tf_gpu_devices(tf_info: dict) -> List[Any]:
    device_list = tf_info.get('tf_device_lib') or []
    return [
        device for device in device_list
        if device.device_type == GPU_DEVICE_TYPE
    ]


def get_gpu_device_name(device: Any) -> str:
    match = GPU_PHYSICAL_DEVICE_DESC_NAME_PATTERN.search(
        getattr(device, 'physical_device_desc', '') or ''
    )
    if match:
        return match.group(1)
    return device.name


def get_tf_device_summary(tf_info: dict) -> str:
    gpu_devices = get_tf_gpu_devices(tf_info)
    if not gpu_devices:
        device_part = 'CPU only'
    else:
        device_part = 'GPU x%d (%s)' % (
            len(gpu_devices),
            ', '.join(get_gpu_device_name(device) for device in gpu_devices)
        )
    return '%s [tf: %s]' % (device_part, tf_info.get('tf_version'))
