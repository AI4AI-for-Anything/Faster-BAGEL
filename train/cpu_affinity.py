"""Optional per-rank CPU affinity; no tensor or training-state changes."""
import json
import os
from pathlib import Path


_configured_pid = None


def rank_cpu_sets(available, local_world_size):
    """Partition allowed physical cores, keeping SMT siblings on one rank."""
    cores = {}
    for cpu in sorted(available):
        topology = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        try:
            key = (int((topology / 'physical_package_id').read_text()),
                   int((topology / 'core_id').read_text()))
        except (OSError, ValueError):
            # Still give each logical CPU exclusively to one rank if the
            # container does not expose physical topology.
            key = (-1, cpu)
        cores.setdefault(key, []).append(cpu)
    groups = [cores[k] for k in sorted(cores)]
    if local_world_size <= 0 or len(groups) < local_world_size:
        raise ValueError('CPU affinity needs at least one allowed core per local rank')
    return [set(cpu for group in groups[i * len(groups) // local_world_size:
                                       (i + 1) * len(groups) // local_world_size]
                for cpu in group) for i in range(local_world_size)]


def configure_cpu_affinity():
    """Bind only this training process and its own existing threads.

    Torchrun provides LOCAL_RANK/LOCAL_WORLD_SIZE. All assignments are subsets
    of this process's original allowed CPUs. New autograd/communication/loader
    threads inherit the assigned mask. OpenMP and PyTorch thread counts stay
    unchanged. Called once per process before FSDP wrapping.
    """
    global _configured_pid
    mode = os.environ.get('BAGEL_OPT_CPU_AFFINITY', '')
    if not mode or _configured_pid == os.getpid():
        return
    if mode != 'compact':
        raise ValueError('BAGEL_OPT_CPU_AFFINITY must be empty or compact')
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['LOCAL_WORLD_SIZE'])
    if not 0 <= rank < size:
        raise ValueError('LOCAL_RANK must be within LOCAL_WORLD_SIZE')
    allowed = os.sched_getaffinity(0)
    selected = rank_cpu_sets(allowed, size)[rank]
    changed = 0
    # /proc/self/task contains only threads of our own process, never another
    # process. Catch a thread exiting during setup.
    for entry in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(entry.name), selected)
            changed += 1
        except ProcessLookupError:
            pass
    _configured_pid = os.getpid()
    print('bagel_cpu_affinity ' + json.dumps({
        'pid': _configured_pid, 'local_rank': rank,
        'cpus': sorted(selected), 'threads_bound': changed,
    }), flush=True)
