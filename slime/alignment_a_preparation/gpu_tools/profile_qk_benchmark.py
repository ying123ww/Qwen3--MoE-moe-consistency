"""Spawn-safe annotation extension around the unchanged benchmark."""
import profiling_annotations
import qk_profile_annotations

profiling_annotations.install = qk_profile_annotations.install

if __name__ == '__main__':
    import gzip
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import sys
    import time
    import inference_benchmark

    # Engine shutdown can race non-primary workers still compressing their traces.
    # Wait outside generate timing, while exporters are alive, before terminating them.
    original_flush = inference_benchmark.flush_cache_until_idle
    output = Path(sys.argv[sys.argv.index('--run') + 1])

    def valid_gzip(path):
        try:
            with gzip.open(path, 'rb') as handle:
                while handle.read(8 * 1024 * 1024):
                    pass
            return True
        except (OSError, EOFError):
            return False

    def flush(engine, *args, **kwargs):
        if not getattr(engine, '_qk_safe_shutdown_installed', False):
            shutdown = engine.shutdown
            def safe_shutdown():
                if (output / 'profiles').exists():
                    deadline = time.monotonic() + 300
                    verified = set()
                    while time.monotonic() < deadline:
                        paths = set(output.glob('profiles/**/*.json.gz'))
                        pending = sorted(paths - verified)
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            verified.update(p for p, ok in zip(pending, pool.map(valid_gzip, pending)) if ok)
                        if len(paths) == len(verified) == 8:
                            print('QK_PROFILE_EXPORTS_VERIFIED 8', flush=True)
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError('Profile exports did not complete before shutdown')
                return shutdown()
            engine.shutdown = safe_shutdown
            engine._qk_safe_shutdown_installed = True
        return original_flush(engine, *args, **kwargs)

    inference_benchmark.flush_cache_until_idle = flush
    inference_benchmark.main()
