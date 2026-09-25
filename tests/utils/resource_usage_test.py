import logging
import time

import pytest
import torch

from sciencebeam_trainer_delft.utils.resource_usage import (
    PhaseTimer,
    get_cpu_seconds,
    get_peak_gpu_memory_mb,
    get_peak_host_memory_mb,
    log_peak_memory_usage
)


RESOURCE_USAGE_LOGGER = 'sciencebeam_trainer_delft.utils.resource_usage'


class TestGetCpuSeconds:
    def test_should_increase_while_work_is_done(self):
        before = get_cpu_seconds()
        assert sum(index * index for index in range(2_000_000)) > 0
        assert get_cpu_seconds() > before

    def test_should_not_go_backwards(self):
        assert get_cpu_seconds() <= get_cpu_seconds()


class TestGetPeakHostMemoryMb:
    def test_should_report_a_plausible_resident_set_size(self):
        # anything importing torch is well past a megabyte and far below a
        # terabyte; the point is that the unit conversion is right
        peak_mb = get_peak_host_memory_mb()
        assert 1 < peak_mb < 1_000_000


class TestGetPeakGpuMemoryMb:
    def test_should_follow_whether_a_device_is_present(self):
        if torch.cuda.is_available():
            assert get_peak_gpu_memory_mb() is not None
        else:
            assert get_peak_gpu_memory_mb() is None


class TestLogPeakMemoryUsage:
    def test_should_log_both_figures(self, caplog):
        with caplog.at_level(logging.INFO, logger=RESOURCE_USAGE_LOGGER):
            log_peak_memory_usage()
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1
        assert 'host rss=' in messages[0]
        assert 'gpu reserved=' in messages[0]


class TestPhaseTimer:
    def test_should_report_cores_as_cpu_time_over_wall_time(self):
        timer = PhaseTimer('example')
        timer.seconds = 2.0
        timer.cpu_seconds = 8.0
        assert timer.cores == 4.0

    def test_should_report_no_cores_for_a_phase_that_did_nothing(self):
        assert PhaseTimer('example').cores == 0

    def test_should_accumulate_over_several_spans(self):
        timer = PhaseTimer('example')
        for _ in range(3):
            with timer:
                time.sleep(0.01)
        assert timer.seconds == pytest.approx(0.03, abs=0.02)

    def test_should_name_the_phase_with_its_seconds_and_cores(self):
        timer = PhaseTimer('batch')
        timer.seconds = 4.0
        timer.cpu_seconds = 6.0
        assert str(timer) == 'batch=4.00s (1.5 cores)'

    def test_should_leave_out_the_cores_of_a_phase_too_short_to_measure(self):
        # dividing two near-zero numbers reports hundreds of cores, so a phase
        # below the kernel's accounting granularity reports its time alone
        timer = PhaseTimer('batch')
        timer.seconds = 0.0001
        timer.cpu_seconds = 0.01
        assert str(timer) == 'batch=0.00s'

    def test_should_refuse_to_stop_without_starting(self):
        with pytest.raises(AssertionError):
            PhaseTimer('example').stop()
