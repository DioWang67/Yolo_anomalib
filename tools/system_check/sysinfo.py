"""Read-only system facts with graceful degradation.

``psutil`` is the preferred source and is bundled with both the application and
the standalone checker, but every accessor here falls back to the standard
library (and, on Windows, to ``ctypes`` against kernel32) so a stripped
environment still produces numbers instead of a traceback.

Nothing in this module identifies the operator or the machine by name. Host
names and user names are deliberately not collected: the report is meant to be
e-mailed between sites.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BYTES_PER_GB = 1024**3
BYTES_PER_MB = 1024**2


def _psutil():
    """Return the ``psutil`` module, or ``None`` when unavailable."""
    try:
        import psutil
    except Exception:  # noqa: BLE001 - a broken install must not abort the run
        return None
    return psutil


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    """Win32 ``MEMORYSTATUSEX``."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class MemoryInfo:
    """Physical memory snapshot in bytes. ``None`` means undetermined."""

    total: int | None
    available: int | None

    @property
    def used(self) -> int | None:
        if self.total is None or self.available is None:
            return None
        return self.total - self.available

    @property
    def percent_used(self) -> float | None:
        if self.total in (None, 0) or self.used is None:
            return None
        return round(self.used / self.total * 100.0, 1)


def memory_info() -> MemoryInfo:
    """Return total and available physical memory."""
    psutil = _psutil()
    if psutil is not None:
        try:
            virtual = psutil.virtual_memory()
            return MemoryInfo(total=int(virtual.total), available=int(virtual.available))
        except Exception:  # noqa: BLE001
            pass
    if os.name == "nt":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return MemoryInfo(total=int(status.ullTotalPhys), available=int(status.ullAvailPhys))
        except Exception:  # noqa: BLE001
            pass
    # POSIX fallback. Reached only when the checker runs off Windows, which the
    # test suite does on CI; ``os.sysconf`` does not exist in the win32 stubs,
    # so it is fetched dynamically rather than called directly.
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return MemoryInfo(total=None, available=None)
    try:
        page_size = sysconf("SC_PAGE_SIZE")
        total = page_size * sysconf("SC_PHYS_PAGES")
        available = page_size * sysconf("SC_AVPHYS_PAGES")
        return MemoryInfo(total=int(total), available=int(available))
    except (AttributeError, ValueError, OSError):
        return MemoryInfo(total=None, available=None)


def process_rss() -> int | None:
    """Return this process's resident set size in bytes, or ``None``."""
    psutil = _psutil()
    if psutil is not None:
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001
            pass
    if os.name == "nt":
        try:

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuInfo:
    """Processor description. ``None`` fields are undetermined."""

    model: str | None
    physical_cores: int | None
    logical_cores: int | None
    max_clock_mhz: float | None
    architecture: str
    flags: tuple[str, ...]


def _windows_cpu_registry() -> tuple[str | None, float | None]:
    """Read the CPU name and rated clock from the Windows registry."""
    if os.name != "nt":
        return None, None
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows only
        return None, None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            try:
                mhz, _ = winreg.QueryValueEx(key, "~MHz")
            except OSError:
                mhz = None
            return str(name).strip(), float(mhz) if mhz is not None else None
    except OSError:
        return None, None


def _cpu_flags() -> tuple[str, ...]:
    """Return CPU feature flags when a source is available.

    ``py-cpuinfo`` ships with ultralytics in the application environment but is
    not bundled with the standalone checker, so an empty tuple is a normal
    outcome and callers must report it as UNKNOWN rather than as a missing
    instruction set.
    """
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
    except Exception:  # noqa: BLE001
        return ()
    flags = info.get("flags") or []
    return tuple(str(flag).lower() for flag in flags)


def cpu_info() -> CpuInfo:
    """Return processor model, core counts, clock and flags."""
    psutil = _psutil()
    physical: int | None = None
    logical: int | None = os.cpu_count()
    clock: float | None = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True) or logical
        except Exception:  # noqa: BLE001
            pass
        try:
            freq = psutil.cpu_freq()
            if freq is not None and freq.max:
                clock = float(freq.max)
        except Exception:  # noqa: BLE001
            pass

    name, registry_clock = _windows_cpu_registry()
    if clock is None:
        clock = registry_clock
    model = name or platform.processor() or None

    return CpuInfo(
        model=model,
        physical_cores=physical,
        logical_cores=logical,
        max_clock_mhz=clock,
        architecture=platform.machine(),
        flags=_cpu_flags(),
    )


def cpu_percent(interval: float = 0.3) -> float | None:
    """Sample system-wide CPU utilisation, or ``None`` when unavailable."""
    psutil = _psutil()
    if psutil is None:
        return None
    try:
        return float(psutil.cpu_percent(interval=interval))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskInfo:
    """Free-space snapshot for one path, in bytes."""

    path: str
    volume: str
    total: int | None
    free: int | None


def disk_info(path: str | Path) -> DiskInfo:
    """Return capacity and free space for the volume holding ``path``.

    Walks up to the nearest existing ancestor, so a result directory that has
    not been created yet still reports its target volume.
    """
    target = Path(path).expanduser()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    volume = str(Path(probe.anchor or probe))
    try:
        usage = shutil.disk_usage(probe)
        return DiskInfo(str(target), volume, int(usage.total), int(usage.free))
    except OSError:
        return DiskInfo(str(target), volume, None, None)


# --------------------------------------------------------------------------
# OS
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OsInfo:
    """Operating system description."""

    system: str
    release: str
    version: str
    build: str | None
    edition: str | None
    machine: str
    is_64bit: bool
    python_64bit: bool


def _windows_edition() -> tuple[str | None, str | None]:
    """Return ``(edition, build)`` from the registry, or ``(None, None)``."""
    if os.name != "nt":
        return None, None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:

            def read(name: str) -> str | None:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                    return str(value)
                except OSError:
                    return None

            build = read("CurrentBuildNumber")
            ubr = read("UBR")
            product = read("ProductName")
            display = read("DisplayVersion")
            full_build = f"{build}.{ubr}" if build and ubr else build
            edition = f"{product} {display}".strip() if product else None
            return edition, full_build
    except (ImportError, OSError):
        return None, None


def os_info() -> OsInfo:
    """Return the operating system description."""
    edition, build = _windows_edition()
    return OsInfo(
        system=platform.system(),
        release=platform.release(),
        version=platform.version(),
        build=build,
        edition=edition,
        machine=platform.machine(),
        is_64bit=platform.machine().lower() in {"amd64", "x86_64", "arm64"},
        python_64bit=sys.maxsize > 2**32,
    )


# --------------------------------------------------------------------------
# Subprocess helper
# --------------------------------------------------------------------------


def run_command(args: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a read-only command with a hard timeout.

    Returns ``(returncode, stdout, stderr)``. A missing executable yields
    ``(-1, "", reason)`` rather than raising, because "``nvidia-smi`` is not
    installed" is a finding the report must carry.
    """
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )
    except FileNotFoundError:
        return -1, "", f"{args[0]} not found on PATH"
    except subprocess.TimeoutExpired:
        return -1, "", f"{args[0]} timed out after {timeout}s"
    except OSError as exc:
        return -1, "", f"{args[0]} failed: {exc}"
    return completed.returncode, completed.stdout or "", completed.stderr or ""
