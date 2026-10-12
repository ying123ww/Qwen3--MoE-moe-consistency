"""Scoped P13 bootstrap: install the validated full-config bypass at lazy import."""
import os
if os.environ.get('SLIME_P13_FULL_BOOTSTRAP')=='1':
    import importlib.abc
    import importlib.machinery
    import sys
    TARGET='sglang.srt.server_args'
    CHECKPOINT_TARGET='torch.distributed.checkpoint.filesystem'
    ENGINE_TARGET='slime.backends.sglang_utils.sglang_engine'
    class FullLoader(importlib.abc.Loader):
        def __init__(self,original):self.original=original
        def create_module(self,spec):
            method=getattr(self.original,'create_module',None)
            return method(spec) if method else None
        def exec_module(self,module):
            self.original.exec_module(module)
            if module.__name__==ENGINE_TARGET:
                import router_ready_guard
                router_ready_guard.install(module)
                return
            if module.__name__==CHECKPOINT_TARGET:
                import checkpoint_read_guard
                checkpoint_read_guard.install(module)
                return
            from pathlib import Path
            from fixed_capacity_experiment import allow_full_configuration
            allow_full_configuration(Path(os.environ['SLIME_P13_RUN_DIR'])/'bootstrap',deterministic=True)
            original=module.ServerArgs._handle_a2a_moe
            def track_endpoint(args):
                original(args)
                from sglang.srt.model_executor.cuda_graph_config import Backend
                if args.cuda_graph_config.decode.backend==Backend.FULL:
                    import json
                    directory=Path(os.environ['SLIME_P13_RUN_DIR'])/'bootstrap';directory.mkdir(parents=True,exist_ok=True)
                    (directory/'server-endpoint.json').write_text(json.dumps(dict(host=args.host,port=args.port,tp_size=args.tp_size,ep_size=args.ep_size))+'\n')
            module.ServerArgs._handle_a2a_moe=track_endpoint
    class FullFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path,target=None):
            if fullname not in (TARGET,CHECKPOINT_TARGET,ENGINE_TARGET):return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is None or spec.loader is None:raise ImportError('P13 ServerArgs implementation missing')
            spec.loader=FullLoader(spec.loader)
            return spec
    sys.meta_path.insert(0,FullFinder())
