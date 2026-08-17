from unittest.mock import patch

import pytest

from sciencebeam_trainer_delft.utils import device as device_module
from sciencebeam_trainer_delft.utils.device import (
    CPU_DEVICE,
    CUDA_DEVICE,
    get_default_device,
    get_device_info,
    get_device_summary,
    get_gpu_device_names,
    get_gpu_devices,
    get_unsupported_gpu_devices,
    log_device_info,
    log_device_info_once,
    validate_device
)


INVALID_DEVICE = 'gpu'


@pytest.fixture(name='reset_log_device_info_once', autouse=True)
def _reset_log_device_info_once():
    # the log is gated once per process, which would otherwise leak between tests
    log_device_info_once.cache_clear()
    yield
    log_device_info_once.cache_clear()


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


class TestWrapperDeviceDefault:
    def test_should_default_to_the_detected_device(self):
        from sciencebeam_trainer_delft.sequence_labelling.wrapper import (  # noqa: E501 pylint: disable=import-outside-toplevel
            get_default_device_or_env
        )
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True):
            assert get_default_device_or_env() == CUDA_DEVICE
        with patch.object(device_module.torch.cuda, 'is_available', return_value=False):
            assert get_default_device_or_env() == CPU_DEVICE

    def test_should_let_the_environment_override_the_detected_device(self, monkeypatch):
        from sciencebeam_trainer_delft.sequence_labelling.wrapper import (  # noqa: E501 pylint: disable=import-outside-toplevel
            EnvironmentVariables,
            get_default_device_or_env
        )
        monkeypatch.setenv(EnvironmentVariables.DEVICE, CPU_DEVICE)
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True):
            assert get_default_device_or_env() == CPU_DEVICE

    def test_should_auto_detect_with_an_empty_environment_value(self, monkeypatch):
        from sciencebeam_trainer_delft.sequence_labelling.wrapper import (  # noqa: E501 pylint: disable=import-outside-toplevel
            EnvironmentVariables,
            get_default_device_or_env
        )
        # the only way to undo a value set by a base image
        monkeypatch.setenv(EnvironmentVariables.DEVICE, '')
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True):
            assert get_default_device_or_env() == CUDA_DEVICE

    def test_should_reject_an_invalid_environment_value(self, monkeypatch):
        from sciencebeam_trainer_delft.sequence_labelling.wrapper import (  # noqa: E501 pylint: disable=import-outside-toplevel
            EnvironmentVariables,
            get_default_device_or_env
        )
        monkeypatch.setenv(EnvironmentVariables.DEVICE, INVALID_DEVICE)
        with pytest.raises(ValueError, match=repr(INVALID_DEVICE)):
            get_default_device_or_env()

    def test_should_name_the_environment_variable_of_an_invalid_value(self, monkeypatch):
        from sciencebeam_trainer_delft.sequence_labelling.wrapper import (  # noqa: E501 pylint: disable=import-outside-toplevel
            EnvironmentVariables,
            get_default_device_or_env
        )
        monkeypatch.setenv(EnvironmentVariables.DEVICE, INVALID_DEVICE)
        with pytest.raises(ValueError, match=EnvironmentVariables.DEVICE):
            get_default_device_or_env()


class TestGetUnsupportedGpuDevices:
    def test_should_report_a_capability_the_wheel_was_not_built_for(self):
        device_info = {'gpu_device_capabilities': ['sm_75'], 'torch_arch_list': ['sm_90']}
        assert get_unsupported_gpu_devices(device_info) == ['sm_75']

    def test_should_report_nothing_when_the_capability_is_built(self):
        device_info = {
            'gpu_device_capabilities': ['sm_75'], 'torch_arch_list': ['sm_75', 'sm_90']
        }
        assert get_unsupported_gpu_devices(device_info) == []

    def test_should_report_nothing_without_an_arch_list(self):
        # a CPU build has no arch list, and no GPU to be incompatible with
        assert get_unsupported_gpu_devices(
            {'gpu_device_capabilities': [], 'torch_arch_list': []}
        ) == []


class TestLogDeviceInfo:
    def test_should_warn_about_an_unsupported_gpu(self, caplog):
        with caplog.at_level('WARNING'):
            log_device_info({
                'torch_version': TORCH_VERSION,
                'gpu_device_capabilities': ['sm_75'],
                'torch_arch_list': ['sm_90']
            })
        assert 'no kernels for sm_75' in caplog.text

    def test_should_not_warn_about_a_supported_gpu(self, caplog):
        with caplog.at_level('WARNING'):
            log_device_info({
                'torch_version': TORCH_VERSION,
                'gpu_device_capabilities': ['sm_75'],
                'torch_arch_list': ['sm_75']
            })
        assert 'no kernels' not in caplog.text


class TestLogDeviceInfoOnce:
    def test_should_log_the_selected_device(self, caplog):
        with caplog.at_level('INFO'):
            log_device_info_once(CPU_DEVICE)
        assert f'using device: {CPU_DEVICE}' in caplog.text

    def test_should_log_the_same_device_once_only(self, caplog):
        with caplog.at_level('INFO'):
            log_device_info_once(CPU_DEVICE)
            log_device_info_once(CPU_DEVICE)
        assert caplog.text.count(f'using device: {CPU_DEVICE}') == 1

    def test_should_log_the_device_info_without_a_selected_device(self, caplog):
        with caplog.at_level('INFO'):
            log_device_info_once()
        assert 'device_info' in caplog.text
        assert 'using device' not in caplog.text

    def test_should_warn_about_cuda_without_a_cuda_device(self, caplog):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=False):
            with caplog.at_level('WARNING'):
                log_device_info_once(CUDA_DEVICE)
        assert 'no CUDA device' in caplog.text

    def test_should_not_warn_about_cuda_with_a_cuda_device(self, caplog):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=True), \
                patch.object(device_module.torch.cuda, 'device_count', return_value=0), \
                patch.object(device_module.torch.cuda, 'get_arch_list', return_value=[]):
            with caplog.at_level('WARNING'):
                log_device_info_once(CUDA_DEVICE)
        assert 'no CUDA device' not in caplog.text

    def test_should_not_warn_about_the_cpu(self, caplog):
        with patch.object(device_module.torch.cuda, 'is_available', return_value=False):
            with caplog.at_level('WARNING'):
                log_device_info_once(CPU_DEVICE)
        assert 'no CUDA device' not in caplog.text


class TestValidateDevice:
    def test_should_return_a_valid_device(self):
        assert validate_device(CPU_DEVICE) == CPU_DEVICE

    def test_should_accept_a_device_index(self):
        assert validate_device(f'{CUDA_DEVICE}:0') == f'{CUDA_DEVICE}:0'

    def test_should_reject_an_invalid_device_naming_the_value(self):
        with pytest.raises(ValueError, match=repr(INVALID_DEVICE)):
            validate_device(INVALID_DEVICE)

    def test_should_name_the_source_of_an_invalid_device(self):
        with pytest.raises(ValueError, match='ENV_VAR_1'):
            validate_device(INVALID_DEVICE, source='ENV_VAR_1')
