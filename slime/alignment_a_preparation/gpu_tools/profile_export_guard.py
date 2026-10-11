"""Wait for eight gzip exporters before shutting down an Engine."""
import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time


def install(benchmark,output):
    original_flush=benchmark.flush_cache_until_idle
    def valid(path):
        try:
            with gzip.open(path,'rb') as handle:
                while handle.read(8*1024*1024):pass
            return True
        except (OSError,EOFError):return False
    def flush(engine,*args,**kwargs):
        if not getattr(engine,'_complete_export_shutdown',False):
            original=engine.shutdown
            def shutdown():
                if (output/'profiles').exists():
                    deadline=time.monotonic()+300;done=set()
                    while time.monotonic()<deadline:
                        paths=set(output.glob('profiles/**/*.json.gz'));pending=sorted(paths-done)
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            done.update(p for p,ok in zip(pending,pool.map(valid,pending)) if ok)
                        if len(paths)==len(done)==8:
                            print('COMPLETE_PROFILE_EXPORTS_VERIFIED 8',flush=True);break
                        time.sleep(2)
                    else:raise RuntimeError('Profiler exporters did not finish before shutdown')
                return original()
            engine.shutdown=shutdown;engine._complete_export_shutdown=True
        return original_flush(engine,*args,**kwargs)
    benchmark.flush_cache_until_idle=flush
