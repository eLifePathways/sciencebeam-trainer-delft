"""What a run actually used, for sizing the machine it asks for.

Separate from `device`, which reports which device a run got rather than what it
did with it.
"""
import logging
import os
import resource
import time
from typing import Optional

import torch

from sciencebeam_trainer_delft.utils.misc import str_to_bool


LOGGER = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
KB_PER_MB = 1024

# below this the kernel's CPU accounting is too coarse for the ratio to mean
# anything, and dividing two near-zero numbers reports hundreds of cores
MIN_MEASURABLE_SECONDS = 0.01

# splitting a step into its parts means waiting for the device at each boundary,
# which serialises work that would otherwise overlap, so it is asked for rather
# than always on
SCIENCEBEAM_DELFT_STEP_TIMING = 'SCIENCEBEAM_DELFT_STEP_TIMING'

# torch.compile traces the model into fused graphs. It costs a long first step
# to compile, and what it does to a given model is not predictable from reading
# it, so it is asked for and measured rather than switched on.
SCIENCEBEAM_DELFT_TORCH_COMPILE = 'SCIENCEBEAM_DELFT_TORCH_COMPILE'

# the number of steps of the first epoch to profile. `step` timing says how long
# forward and backward take but not what is inside them, and backward has no
# phases to instrument, because autograd does not run the model's own code.
SCIENCEBEAM_DELFT_PROFILE_STEPS = 'SCIENCEBEAM_DELFT_PROFILE_STEPS'
DEFAULT_PROFILE_ROWS = 15


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

    def __init__(self, name: str, enabled: bool = True, synchronize: bool = False):
        self.name = name
        # a timer that is switched off costs an attribute check per span, so a
        # caller can time unconditionally without branching around it
        self.enabled = enabled
        # wait for the device before reading the clock, where a phase boundary
        # would otherwise fall before the work it is meant to cover
        self.synchronize = synchronize
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

    def reset(self) -> 'PhaseTimer':
        self.seconds = 0.0
        self.cpu_seconds = 0.0
        return self

    def start(self) -> 'PhaseTimer':
        if not self.enabled:
            return self
        if self.synchronize:
            synchronize_device()
        self._wall_start = time.perf_counter()
        self._cpu_start = get_cpu_seconds()
        return self

    def stop(self) -> 'PhaseTimer':
        if not self.enabled:
            return self
        assert self._wall_start is not None, 'stop() without start()'
        assert self._cpu_start is not None
        if self.synchronize:
            synchronize_device()
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


def is_step_timing_enabled() -> bool:
    """Reports whether the training step should be timed part by part."""
    return bool(str_to_bool(
        os.environ.get(SCIENCEBEAM_DELFT_STEP_TIMING, ''), default_value=False
    ))


def synchronize_device() -> None:
    """Waits for the device, so that a phase boundary means what it says.

    Device work is queued rather than run, so without this a phase ends when its
    kernels were submitted and the time they take is charged to whichever phase
    happens to wait for them.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def is_torch_compile_enabled() -> bool:
    """Reports whether the model should be handed to `torch.compile`."""
    return bool(str_to_bool(
        os.environ.get(SCIENCEBEAM_DELFT_TORCH_COMPILE, ''), default_value=False
    ))


def get_profile_steps() -> int:
    """Returns how many steps of the first epoch to profile, zero for none."""
    value = os.environ.get(SCIENCEBEAM_DELFT_PROFILE_STEPS, '').strip()
    if not value:
        return 0
    steps = int(value)
    if steps < 0:
        raise ValueError(
            '%s must not be negative: %r' % (SCIENCEBEAM_DELFT_PROFILE_STEPS, value)
        )
    return steps


def log_profile_table(profiler, rows: int = DEFAULT_PROFILE_ROWS) -> None:
    """Logs what the profiler recorded, ordered by the time spent in each operator.

    Sorted by device time where there is a device, because that is the half the
    phase timings cannot reach: `backward` runs in autograd rather than in the
    model, so it has no phases of its own to measure and only an operator
    breakdown says what it is made of.
    """
    sort_by = (
        'self_device_time_total' if torch.cuda.is_available() else 'self_cpu_time_total'
    )
    table = profiler.key_averages().table(sort_by=sort_by, row_limit=rows)
    LOGGER.info('profile, by %s:\n%s', sort_by, table)
