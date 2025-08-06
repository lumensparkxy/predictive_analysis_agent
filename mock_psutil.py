"""
Mock psutil module for testing performance profiling without external dependencies.
"""

import time
import random
from collections import namedtuple

# Mock memory info
MemoryInfo = namedtuple('MemoryInfo', ['rss', 'vms'])
VirtualMemory = namedtuple('VirtualMemory', ['total', 'available', 'percent', 'used', 'free'])
DiskIO = namedtuple('DiskIO', ['read_count', 'write_count', 'read_bytes', 'write_bytes', 'read_time', 'write_time'])
NetworkIO = namedtuple('NetworkIO', ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv'])


class MockProcess:
    """Mock process object."""
    
    def __init__(self, pid=None):
        self.pid = pid or 1234
        self._memory_base = random.randint(100, 500) * 1024 * 1024  # 100-500 MB
    
    def memory_info(self):
        # Simulate slight memory variations
        rss = self._memory_base + random.randint(-10, 10) * 1024 * 1024
        return MemoryInfo(rss=rss, vms=rss * 2)


def cpu_percent(interval=None, percpu=False):
    """Mock CPU percentage."""
    if percpu:
        return [random.uniform(5, 95) for _ in range(4)]  # Mock 4 cores
    return random.uniform(10, 85)


def virtual_memory():
    """Mock virtual memory statistics."""
    total = 8 * 1024 * 1024 * 1024  # 8GB
    used = random.randint(2, 6) * 1024 * 1024 * 1024  # 2-6GB used
    available = total - used
    percent = (used / total) * 100
    return VirtualMemory(
        total=total,
        available=available,
        percent=percent,
        used=used,
        free=available
    )


def disk_io_counters():
    """Mock disk I/O counters."""
    base_time = int(time.time())
    return DiskIO(
        read_count=random.randint(1000, 5000),
        write_count=random.randint(500, 2000),
        read_bytes=random.randint(100, 1000) * 1024 * 1024,  # MB
        write_bytes=random.randint(50, 500) * 1024 * 1024,   # MB
        read_time=random.randint(100, 1000),
        write_time=random.randint(50, 500)
    )


def net_io_counters():
    """Mock network I/O counters."""
    return NetworkIO(
        bytes_sent=random.randint(100, 1000) * 1024 * 1024,    # MB
        bytes_recv=random.randint(200, 2000) * 1024 * 1024,    # MB
        packets_sent=random.randint(10000, 50000),
        packets_recv=random.randint(20000, 100000)
    )


def getloadavg():
    """Mock load average (Unix-like systems only)."""
    return [random.uniform(0.1, 4.0) for _ in range(3)]


def pids():
    """Mock process IDs list."""
    return list(range(1, random.randint(150, 300)))


def cpu_count():
    """Mock CPU count."""
    return 4


def Process(pid=None):
    """Mock Process constructor."""
    return MockProcess(pid)