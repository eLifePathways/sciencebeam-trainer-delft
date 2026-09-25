"""What a run actually used, for sizing the machine it asks for.

Separate from `device`, which reports which device a run got rather than what it
did with it.
"""
import logging
import resource
import time
from typing import Optional

import torch


LOGGER = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
KB_PER_MB = 1024

# below this the kernel's CPU accounting is too coarse for the ratio to mean
# anything, and dividing two near-zero numbers reports hundreds of cores
MIN_MEASURABLE_SECONDS = 0.01


def get_cpu_seconds() -> float:
    """Returns the CPU seconds used so far, across this process's threads.

    Children are included, but only once they have exited and been reaped --
    that is what `RUSAGE_CHILDREN` reports, and `os.times` is no different. So
    this is accurate for the threaded work of today, and would *under*-report
    live worker processes: their time would appear all at once when they
    finished rather than accruing per epoch. Measuring those needs their CPU
    time read while they run, per pid, which is worth adding with the workers
    rather than before them.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    finished_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        usage.ru_utime + usage.ru_stime
        + finished_children.ru_utime + finished_children.ru_stime
    )


def get_peak_host_memory_mb() -> float:
    """Returns the peak resident set size of this process, in MB."""
    # ru_maxrss is in kilobytes on Linux, which is where this runs
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB_PER_MB


def get_peak_gpu_memory_mb() -> Optional[float]:
    """Returns the peak memory torch reserved on the device, in MB, if any."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / BYTES_PER_MB


def log_peak_memory_usage() -> None:
    """Logs what the run needed, which is what sizing a machine turns on.

    The device figure is what torch reserved, which tracks `nvidia-smi` more
    closely than what it allocated does. Neither includes the CUDA context, so
    it is a floor for the card rather than the whole of what one needs.
    """
    peak_gpu_memory_mb = get_peak_gpu_memory_mb()
    LOGGER.info(
        'peak memory: host rss=%.0f MB, gpu reserved=%s',
        get_peak_host_memory_mb(),
        'n/a' if peak_gpu_memory_mb is None else '%.0f MB' % peak_gpu_memory_mb
    )


class PhaseTimer:
    """Accumulates the wall and CPU time of one phase, over any number of spans.

    Reporting both is what makes the figure readable: wall time alone cannot
    distinguish a phase that saturates the machine from one that leaves it
    idle, and CPU time alone cannot say how long anyone waited. Their ratio is
    the average number of cores the phase kept busy.

    The CPU side is the whole process, so a phase is credited with any thread
    running during its window rather than only its own work. Where the phases
    run one after another, as they do here, that is what is wanted; it would
    mislead if two were timed at once, and torch's pools can spin between
    operations, which lifts the count of a phase that does not itself use them.
    """

    def __init__(self, name: str):
        self.name = name
        self.seconds = 0.0
        self.cpu_seconds = 0.0
        self._wall_start: Optional[float] = None
        self._cpu_start: Optional[float] = None

    @property
    def cores(self) -> float:
        """The average cores busy: one for a single thread, zero while waiting."""
        if self.seconds <= 0:
            return 0.0
        return self.cpu_seconds / self.seconds

    def start(self) -> 'PhaseTimer':
        self._wall_start = time.perf_counter()
        self._cpu_start = get_cpu_seconds()
        return self

    def stop(self) -> 'PhaseTimer':
        assert self._wall_start is not None, 'stop() without start()'
        assert self._cpu_start is not None
        self.seconds += time.perf_counter() - self._wall_start
        self.cpu_seconds += get_cpu_seconds() - self._cpu_start
        self._wall_start = None
        self._cpu_start = None
        return self

    def __enter__(self) -> 'PhaseTimer':
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __str__(self) -> str:
        if self.seconds < MIN_MEASURABLE_SECONDS:
            return '%s=%.2fs' % (self.name, self.seconds)
        return '%s=%.2fs (%.1f cores)' % (self.name, self.seconds, self.cores)
