import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import run_full_graph_profile as runner
folder=Path(__file__).resolve().parent
paths=[Path(__file__),folder/'raw_projection_hooks.py',folder/'profile_raw_benchmark.py']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
original_command=runner.guard.command

def command(run,label,argv,env,*args,**kwargs):
    if label=='C':
        argv=[str(folder/'profile_raw_benchmark.py') if x.endswith('/full_graph_profile_benchmark.py') else x for x in argv]
        env=dict(env);env['PYTHONPATH']=str(folder)+':'+env['PYTHONPATH']
        runner.guard.save(run/'raw-source-sha256.json',hashes)
        for source in paths:
            (run/source.name).write_bytes(source.read_bytes())
    return original_command(run,label,argv,env,*args,**kwargs)
runner.guard.command=command
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='P11 C full graph raw projections, dumped immediately per bucket')
    parser.add_argument('--phase',choices=('smoke',),default='smoke')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--p10-run',type=Path,required=True)
    args=parser.parse_args()
    sys.argv=[sys.argv[0],'--phase','smoke','--run-id',args.run_id,'--p10-run',str(args.p10_run)]
    runner.main()
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
