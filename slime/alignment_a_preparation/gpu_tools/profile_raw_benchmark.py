import asyncio
from pathlib import Path
import full_graph_profile_benchmark as benchmark
from raw_projection_hooks import KEY
original_configure=benchmark.experiment.configure

def raw_dump(engine):
    from sglang.srt.managers.io_struct import SetInternalStateReq
    replies=engine.loop.run_until_complete(asyncio.wait_for(engine.tokenizer_manager.set_internal_state(SetInternalStateReq(server_args={KEY:1})),timeout=120))
    if not replies or not all(replies):raise RuntimeError('Raw dump control not acknowledged')

def configure(config,mode,evidence,*args,**kwargs):
    if mode!='C':raise ValueError('Raw projection probe is alignment only')
    result=original_configure(config,mode,evidence,*args,**kwargs)
    from sglang.srt.entrypoints.engine import Engine
    original_generate=Engine.generate
    def generate(engine,*generate_args,**generate_kwargs):
        output=original_generate(engine,*generate_args,**generate_kwargs)
        if len(generate_kwargs.get('input_ids',[]))==3 and generate_kwargs.get('sampling_params',{}).get('max_new_tokens')==2:
            # Dump bucket2 immediately; replaying another graph sharing its pool
            # can overwrite inactive graph tensors even if references are kept.
            raw_dump(engine)
        return output
    Engine.generate=generate
    result['forward_hooks'].append(dict(name='raw-original-projections',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='raw_projection_hooks:install',config=dict(output_dir=str(Path(evidence).parent/'raw-projections'))))
    return result
benchmark.experiment.configure=configure
original_command=benchmark.command

def command(engine,value):
    original_command(engine,value)
    if value==2:raw_dump(engine)
benchmark.command=command
if __name__=='__main__':benchmark.main()
