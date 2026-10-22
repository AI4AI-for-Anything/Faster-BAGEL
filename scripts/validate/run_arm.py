"""Run one 2000-step production arm, preserving checkpoints and recording its outcome."""
import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tree',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,choices=[4396,4397,4398,4399],required=True);p.add_argument('--reference',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();assert not out.exists(),'Use a fresh output directory; do not overwrite a run'
for key in ['CHECKPOINT_DIR','CACHE_DIR']:
 path=Path(os.environ[key]).resolve();assert not path.exists(),f'{key} must be fresh'
out.mkdir(parents=True);env=os.environ.copy()
env.update(CODE_TREE=str(Path(a.tree).resolve()),OUTPUT_DIR=str(out),GLOBAL_SEED=str(a.seed),TOTAL_STEPS='2000',REFERENCE='1' if a.reference else '0')
start=time.time();meta={'seed':a.seed,'data_seed':42,'steps':2000,'production_mode':True,'reference':a.reference,'wrapper_pid':os.getpid(),'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
child=subprocess.Popen(['bash',str(r/'scripts/train_faster_bagel_sft.sh')],env=env,start_new_session=True)
meta.update(pid=child.pid,process_group=child.pid);(out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
try:rc=child.wait(timeout=7200)
except subprocess.TimeoutExpired:
 meta['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
 try:rc=child.wait(timeout=120)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
meta.update(exit_code=rc,wall_s=time.time()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta));raise SystemExit(rc)
