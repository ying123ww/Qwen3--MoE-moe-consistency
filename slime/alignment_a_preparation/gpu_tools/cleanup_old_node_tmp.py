import os,json,shutil,socket
from pathlib import Path
assert socket.gethostname().split('.')[0]=='ai_hgx_03'
uid=os.getuid()
roots=[Path('/tmp/ray-alignment-a-20260909'),Path('/tmp/ray-v41')]
roots+=list(Path('/tmp').glob('sglang_temp_file_17889*.sh'))
roots+=list(Path('/tmp').glob('wandb_gpu_stats-*.sock'))
roots += [p for p in Path('/tmp').glob('tmp*wandb-*') if p.is_dir() and not any(p.iterdir())]
refs=[]
cmds=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=uid:continue
  cmds.append((proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
  for f in [proc/'cwd',proc/'exe',*list((proc/'fd').iterdir())]:
   try:refs.append(os.readlink(f))
   except OSError:pass
  refs.extend((proc/'maps').read_text().splitlines())
 except (OSError,PermissionError):continue
rows=[]
for p in roots:
 if not p.exists() and not p.is_socket():continue
 if p.lstat().st_uid!=uid or p.is_symlink():continue
 reason=None
 if any(str(p) in r for r in refs+cmds):reason='in-use reference'
 if p.name.startswith('wandb_gpu_stats-'):
  pid=p.name.split('-')[1]
  if Path('/proc',pid).exists():reason='producer PID exists'
 if reason:
  rows.append(dict(path=str(p),action='keep',reason=reason));continue
 size=sum(q.lstat().st_blocks*512 for q in p.rglob('*') if not q.is_symlink()) if p.is_dir() else p.lstat().st_blocks*512
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
 rows.append(dict(path=str(p),action='removed',allocated_bytes=size))
out=Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/gpu_runs/node-tmp-cleanup-20260910.json')
out.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
print('freed allocated bytes',sum(r.get('allocated_bytes',0) for r in rows))
