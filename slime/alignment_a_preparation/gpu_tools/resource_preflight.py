"""Read actual host/cgroup limits before importing the training stack."""
import math
import os
from pathlib import Path
import shutil

GIB = 1024 ** 3


def cgroup_dirs():
    memberships = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
    result = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ('cgroup', 'cgroup2'):
            continue
        mount_root, mountpoint = fields[3:5]
        for _, controllers, member in memberships:
            if fs[0] == 'cgroup2' and controllers:
                continue
            if fs[0] == 'cgroup' and not set(controllers.split(',')).intersection(fs[2].split(',')):
                continue
            if member != mount_root and not member.startswith(mount_root.rstrip('/') + '/'):
                continue
            relative = member[len(mount_root):].lstrip('/')
            current, stop = Path(mountpoint) / relative, Path(mountpoint)
            while True:
                if current not in result:
                    result.append(current)
                if current == stop:
                    break
                current = current.parent
    return result


def snapshot(temp_dir):
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    affinity = sorted(os.sched_getaffinity(0))
    cpus = len(affinity)
    available = memory['MemAvailable']
    groups = []
    memory_seen = False
    for directory in cgroup_dirs():
        values = {}
        for name in ('memory.max', 'memory.current', 'memory.events', 'cpu.max',
                     'cpuset.cpus.effective', 'cpuset.mems.effective',
                     'memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.failcnt',
                     'cpu.cfs_quota_us', 'cpu.cfs_period_us'):
            p = directory / name
            if p.exists():
                values[name] = p.read_text().strip()
        for limit, usage in (('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            if limit in values and usage in values:
                memory_seen = True
                if values[limit] != 'max':
                    available = min(available, max(0, int(values[limit]) - int(values[usage])))
        if 'cpu.max' in values:
            quota, period = values['cpu.max'].split()
            if quota != 'max':
                cpus = min(cpus, math.floor(int(quota) / int(period)))
        if 'cpu.cfs_quota_us' in values and int(values['cpu.cfs_quota_us']) > 0:
            cpus = min(cpus, math.floor(int(values['cpu.cfs_quota_us']) / int(values['cpu.cfs_period_us'])))
        groups.append({'path': str(directory), 'values': values})
    parent = Path(temp_dir)
    while not parent.exists():
        parent = parent.parent
    return {'cpu_affinity': affinity, 'ray_cpus': cpus, 'host_memory': memory,
            'effective_memory_available_bytes': available, 'cgroup_memory_readable': memory_seen,
            'cgroups': groups, 'temp_free_bytes': shutil.disk_usage(parent).free,
            'shm_free_bytes': shutil.disk_usage('/dev/shm').free,
            'pressure': {name: Path('/proc/pressure', name).read_text() for name in ('cpu', 'memory', 'io')
                         if Path('/proc/pressure', name).exists()}}


def validate(state, min_memory_gib, min_disk_gib):
    errors = []
    if not state['cgroup_memory_readable']:
        errors.append('Cannot establish cgroup memory limits')
    if state['ray_cpus'] < 8:
        errors.append('Fewer than 8 effective CPUs for eight training ranks')
    if state['effective_memory_available_bytes'] < min_memory_gib * GIB:
        errors.append('Insufficient host/cgroup memory headroom')
    if state['temp_free_bytes'] < min_disk_gib * GIB:
        errors.append('Insufficient Ray temporary disk space')
    if state['shm_free_bytes'] < 8 * GIB:
        errors.append('Insufficient /dev/shm for bounded object store')
    if errors:
        raise RuntimeError('; '.join(errors))
