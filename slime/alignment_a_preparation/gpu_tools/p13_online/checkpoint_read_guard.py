"""Task-local serialized DCP file reads; model tensors and training math unchanged."""
import ctypes
import fcntl
import functools
import json
import os
from pathlib import Path
import time


def install(module):
    original=module.FileSystemReader.read_data
    @functools.wraps(original)
    def read(self,*args,**kwargs):
        if os.environ.get('SLIME_P13_SERIAL_CHECKPOINT_READ')!='1':return original(self,*args,**kwargs)
        run=Path(os.environ['SLIME_P13_RUN_DIR']);directory=run/'checkpoint-memory';directory.mkdir(parents=True,exist_ok=True)
        def rss():
            return next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
        before=rss();started=time.monotonic()
        with open(os.environ['SLIME_P13_CHECKPOINT_LOCK'],'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            acquired=time.monotonic();result=original(self,*args,**kwargs)
            if hasattr(result,'wait'):result.wait()
            libc=ctypes.CDLL(None)
            if hasattr(libc,'malloc_trim'):libc.malloc_trim(0)
            after=rss()
        row=dict(pid=os.getpid(),time=time.time(),rss_before=before,rss_after=after,wait_s=acquired-started,read_s=time.monotonic()-acquired,scope='Serialized task-local file read; same checkpoint tensors copied by original reader')
        with (directory/f'read-pid-{os.getpid()}.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        return result
    module.FileSystemReader.read_data=read
