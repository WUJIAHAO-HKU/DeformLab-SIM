"""Per-frame snapshots only; no tracing/profiling or per-operator hooks."""
import ctypes,os,resource,time
from pathlib import Path

_libc=ctypes.CDLL(None);_getcpu=_libc.sched_getcpu;_getcpu.restype=ctypes.c_int
def snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF);cpu=_getcpu()
    return dict(unix=time.time(),monotonic=time.monotonic(),cpu=cpu,user_s=usage.ru_utime,system_s=usage.ru_stime,minor_faults=usage.ru_minflt,major_faults=usage.ru_majflt,voluntary_switches=usage.ru_nvcsw,involuntary_switches=usage.ru_nivcsw)

def difference(before,after):
    return dict(before=before,after=after,delta={k:after[k]-before[k]for k in ['user_s','system_s','minor_faults','major_faults','voluntary_switches','involuntary_switches']})
