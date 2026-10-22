"""Recompute both fixed analyses from raw training and held-out records, without GPUs."""
import argparse,json,math
from pathlib import Path
from verdict import compare,training_result,mean,finite

def read(p):return json.loads(p.read_text())
def old_compare(v):
 if not all(finite(x) for x in v.values()):return {'values':v,'defined':False,'consistent':False}
 y=v['ref_s2']-v['ref_s1'];e1=v['opt_s1']-v['ref_s1'];e2=v['opt_s2']-v['ref_s2'];s=v['opt_s2']-v['opt_s1']
 return {'values':v,'defined':True,'Y':y,'E1':e1,'E2':e2,'S':s,'consistent':all(abs(e)<=abs(y) for e in [e1,e2,s])}
def analyze(records):
 arms=[f'{k}_s{i}' for k in ['ref','opt'] for i in range(1,5)];raw={};held={};stable=True;spikes={}
 for a in arms:
  d=records/a;rows=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()];raw[a]=rows
  assert [r['step'] for r in rows]==list(range(2000)),f'Incomplete {a}; preserve and report failure, do not rerun'
  assert read(d/'run.json')['exit_code']==0,f'Failed {a}; preserve and report'
  stable &= all(finite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
  spikes[a]={'events':[{'step':r['step'],'norm':r['total_norm']} for r in rows if r['total_norm']>10],'maximum':max(r['total_norm'] for r in rows if finite(r['total_norm']))}
  held[a]={}
  for k in ['model','ema']:
   e=read(d/f'heldout_{k}/result.json');assert e['rows']==100 and e['total_ce_tokens']==8407
   assert [x['row'] for x in e['per_row']]==list(range(100))
   stable &= e['all_finite'];held[a][k]=e['mean_text_ce']
   if e['all_finite']:assert math.isclose(math.fsum(x['loss_sum'] for x in e['per_row'])/8407,e['mean_text_ce'],abs_tol=1e-14)
 first={};pooled={}
 for metric in ['ce','mse','log_grad_norm']:
  ww=[];old=[]
  for start in range(0,2000,100):
   values={a:mean([math.log(r['total_norm']) if finite(r['total_norm']) and r['total_norm']>0 else None for r in rows[start:start+100]]) if metric=='log_grad_norm' else mean([r[metric] for r in rows[start:start+100]]) for a,rows in raw.items()}
   w=compare([values[f'ref_s{i}'] for i in range(1,5)],[values[f'opt_s{i}'] for i in range(1,5)]);w.update(start_step=start,end_step=start+99);ww.append(w)
   o=old_compare({a:v for a,v in values.items() if a.endswith(('1','2'))});o.update(start_step=start,end_step=start+99);old.append(o)
  first[metric]=old;pooled[metric]=training_result(ww)
 first_held={k:old_compare({a:e[k] for a,e in held.items() if a.endswith(('1','2'))}) for k in ['model','ema']}
 pooled_held={k:compare([held[f'ref_s{i}'][k] for i in range(1,5)],[held[f'opt_s{i}'][k] for i in range(1,5)]) for k in ['model','ema']}
 first_ok=stable and all(w['consistent'] for ws in first.values() for w in ws) and all(x['consistent'] for x in first_held.values())
 pooled_ok=stable and all(t['passes'] for t in pooled.values()) and all(x['consistent'] for x in pooled_held.values())
 return {'round1':{'verdict':'CONSISTENT' if first_ok else 'NOT CONSISTENT','windows':first,'heldout':first_held},'round2':{'verdict':'CONSISTENT' if pooled_ok else 'NOT CONSISTENT','training':pooled,'heldout':pooled_held},'all_finite':stable,'gradient_spikes':spikes},raw

def main():
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-archive',type=Path);a=p.parse_args()
 result,raw=analyze(a.records)
 if a.verify_archive:
  one=read(a.verify_archive/'summary.json');two=read(a.verify_archive/'round2/summary.json')
  assert result['round1']['verdict'].lower().replace(' ','_')==one['fixed_criterion_verdict']
  assert result['round2']['verdict']==two['pooled_verdict']
  for k,ws in result['round1']['windows'].items():
   for x,y in zip(ws,one['windows'][k]):
    assert x['consistent']==y['consistent']
    for key in ['Y','E1','E2','S']:assert math.isclose(x[key],y[key],rel_tol=1e-12,abs_tol=1e-14)
  for k,t in result['round2']['training'].items():
   saved=two['pooled_training'][k];assert t==saved,(k,'pooled archive mismatch')
  assert result['round2']['heldout']==two['pooled_heldout']
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 lines=['# Recomputed validation',f'Round 1: **{result["round1"]["verdict"]}**. Round 2: **{result["round2"]["verdict"]}**. All finite: {result["all_finite"]}.','Round 1 requires |E1|, |E2| and |S| ≤ |Y| at every compared value. Round 2 uses |D| ≤ 2 × reference sample SD (ddof=1), at most 2/20 failures and a passing final window, plus both held-out metrics. Paired t p and sign-test flags are descriptive.']
 for metric,t in result['round2']['training'].items():
  lines += [f'## {metric}',f'Failed windows: {t["failed_windows"]}/20; final: {t["final_window_pass"]}; bias flag: {t["bias"]["flag_p_lt_0_01"]}; binomial p: {t["bias"]["two_sided_binomial_p"]}.','| Start | D | δ | Paired t p | Consistent |','|---:|---:|---:|---:|:---:|']
  lines += [f'| {w["start_step"]} | {w["D"]} | {w["delta"]} | {w["paired_t_p"]} | {w["consistent"]} |' for w in t['windows']]
 lines += ['## Held-out model and EMA','```json',json.dumps(result['round2']['heldout'],indent=2),'```']
 (a.output/'verdict.md').write_text('\n\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for ax,metric in zip(axs,['ce','mse']):
  for kind,color in [('ref','#A99A9C'),('opt','#8C0000')]:
   curves=np.array([[mean([r[metric] for r in raw[f'{kind}_s{i}'][j:j+10]]) for j in range(0,2000,10)] for i in range(1,5)],dtype=float)
   avg=curves.mean(axis=0);sd=curves.std(axis=0,ddof=1);x=np.arange(200)*10+4.5
   ax.plot(x,avg,label=kind,color=color);ax.fill_between(x,avg-sd,avg+sd,color=color,alpha=.2)
  ax.set(xlabel='Step',ylabel=metric.upper(),title='Four-seed mean ± sample SD');ax.legend()
 fig.savefig(a.output/'loss_curves.png',dpi=160);plt.close(fig)
 print(json.dumps({'round1':result['round1']['verdict'],'round2':result['round2']['verdict'],'all_finite':result['all_finite']}))
if __name__=='__main__':main()
