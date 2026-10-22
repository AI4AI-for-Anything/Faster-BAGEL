"""Render archived measurements in PKU crimson; no GPU operations."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent
RED='#8C0000';LIGHT='#D7BDBF';GREY='#A99A9C';INK='#292122';GRID='#EADFE0'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'faster-bagel-crimson','font.size':11})
p=argparse.ArgumentParser();p.add_argument('--preview-dir',type=Path);args=p.parse_args()
def read(name):return json.loads((R/name).read_text())
def save(fig,name):
 fig.savefig(R/(name+'.svg'),facecolor='white',metadata={'Date':None})
 if args.preview_dir:
  args.preview_dir.mkdir(parents=True,exist_ok=True);fig.savefig(args.preview_dir/(name+'.png'),dpi=100)
 plt.close(fig)
def frame(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.5,color=GRID);ax.set_axisbelow(True)
h=read('history.json')['rows'];m=h[5]
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.barh(['M6','Reference'],[m['candidate_s'],m['reference_s']],color=[RED,LIGHT],height=.45)
for i,v in enumerate([m['candidate_s'],m['reference_s']]):ax.text(v+.025,i,f'{v:.4f} s',va='center',color=INK)
ax.set(xlim=(0,1.8),xlabel='Seconds per training step',title='Faster-BAGEL • 1.8249× • full check PASS (N1)');frame(ax)
fig.suptitle('8 × 80 GB Hopper GPUs · official checkpoint SFT · FP32 state / BF16 compute',fontsize=11);save(fig,'01-benchmark')
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');xx=np.arange(6);yy=[x['speedup'] for x in h[:6]];ax.plot(xx,yy,'o-',color=RED,lw=2)
for i,v in enumerate(yy):ax.annotate(f'{v:.4f}×',(i,v),xytext=(0,12),textcoords='offset points',ha='center')
ax.scatter([6],[h[6]['speedup']],marker='x',s=100,color=INK);ax.annotate('1.8972× timing\nINCORRECT; rejected',(6,h[6]['speedup']),xytext=(-10,20),textcoords='offset points',ha='right')
ax.set(xticks=range(7),xticklabels=[x['milestone'] for x in h],ylim=(1,2.2),ylabel='Cumulative speedup',title='Agent optimization history — M6 retained; 3× goal not reached');frame(ax)
fig.text(.2,.01,'Full checks of cumulative stacks; increments are not isolated switch ablations.',fontsize=9);fig.get_layout_engine().set(rect=(0,.07,1,.93));save(fig,'02-history')
b=read('time-budget.json');fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
parts=[b['compute_union_s_per_step']*1000,b['nccl_exposed_s_per_step']*1000,(b['s_per_step']-b['compute_union_s_per_step']-b['nccl_exposed_s_per_step'])*1000]
axs[0].barh(['Compute union','Exposed NCCL','Remaining span'],parts,color=[RED,LIGHT,GREY]);axs[0].set(xlabel='Profiled ms / step',title='Retained M6: 914.9 ms profiled step')
for i,v in enumerate(parts):axs[0].text(v+5,i,f'{v:.1f}',va='center')
axs[0].set_xlim(0,760)
cats={k:v['s_per_step']*1000 for k,v in b['categories'].items() if k!='nccl'};axs[1].barh(list(cats),list(cats.values()),color=RED);axs[1].set(xlabel='Sum of kernel durations (ms)',title='Compute categories; durations may overlap')
for ax in axs:frame(ax)
fig.suptitle('Latest retained-stack profile • rank 0 • not a performance benchmark',color=RED)
fig.text(.07,.01,'Earlier compute-only estimate ≈2.1×; packed max/mean load 1.24; GEMMs ≈51% of peak. Bounds are estimates.',fontsize=9);fig.get_layout_engine().set(rect=(0,.07,1,.93));save(fig,'03-time-budget')
v=read('validation.json');fig,axs=plt.subplots(5,2,figsize=(14,21),layout='constrained');xs=np.arange(200)*10+4.5
for row,metric in enumerate(['ce','mse']):
 ax=axs[row,0]
 for a in ['ref_s1','ref_s2','opt_s1','opt_s2']:
  ax.plot(xs,v['curves'][a][metric],label=a,color=GREY if a.startswith('ref') else RED,linestyle='-' if a.endswith('1') else '--',lw=1.5)
 ax.set(title=f'Round 1 — {metric.upper()}, two seeds per arm',ylabel=metric.upper());ax.legend(fontsize=9)
 ws=v['round1']['windows'][metric];wx=[w['start_step']+49.5 for w in ws];ax=axs[row,1];y=np.array([abs(w['Y']) for w in ws]);ax.fill_between(wx,-y,y,color=LIGHT,alpha=.6,label='±|Y|')
 for key,style in [('E1','-'),('E2','--'),('S',':')]:ax.plot(wx,[w[key] for w in ws],style,label=key,lw=1.5)
 ax.axhline(0,color=INK,lw=.5);ax.set(title=f'Round 1 {metric.upper()} differences — NOT CONSISTENT',ylabel='Mean difference');ax.legend(fontsize=9)
for col,k in enumerate(['model','ema']):
 ax=axs[2,col]
 for kind,offset,c in [('ref',-.08,GREY),('opt',.08,RED)]:ax.scatter(np.arange(4)+offset,[v['heldout'][f'{kind}_s{i}'][k] for i in range(1,5)],label=kind,color=c,s=45)
 w=v['round2']['heldout'][k];ax.set(xticks=range(4),xticklabels=['4396','4397','4398','4399'],xlabel='Global seed',ylabel='Token-weighted CE',title=f'Held-out {k}: pooled consistent; round-1 criterion fails')
 ax.legend(fontsize=9);ax.text(.03,.04,f'D={w["D"]:.6g}, δ={w["delta"]:.6g}',transform=ax.transAxes)
for col,metric in enumerate(['ce','mse']):
 ax=axs[3,col]
 for kind,color in [('ref',GREY),('opt',RED)]:
  mat=np.array([v['curves'][f'{kind}_s{i}'][metric] for i in range(1,5)]);avg=mat.mean(0);sd=mat.std(0,ddof=1)
  ax.plot(xs,avg,color=color,label=kind);ax.fill_between(xs,avg-sd,avg+sd,color=color,alpha=.25)
 ax.set(title=f'Round 2 {metric.upper()}: four-seed mean ± sample SD',ylabel=metric.upper());ax.legend(fontsize=9)
 ws=v['round2']['training'][metric]['windows'];x=np.array([w['start_step']+49.5 for w in ws]);d=np.array([w['D'] for w in ws]);delta=np.array([w['delta'] for w in ws]);ax=axs[4,col]
 ax.fill_between(x,-delta,delta,color=LIGHT,alpha=.6,label='±δ');ax.plot(x,d,'o-',color=RED,label='D');bad=np.array([not w['consistent'] for w in ws]);ax.scatter(x[bad],d[bad],marker='x',s=70,color=INK,label='Failed window')
 ax.set(title=f'Round 2 {metric.upper()}: pooled criterion passes',ylabel='Mean difference');ax.legend(fontsize=9)
for i,row in enumerate(axs):
 for ax in row:
  frame(ax)
  if i!=2:ax.set_xlabel('Training step')
fig.suptitle('Full-length validation — round 1 NOT CONSISTENT; preregistered round 2 CONSISTENT\nAll eight runs finite • 2000 steps each • disjoint VLM held-out only',color=RED,fontsize=17);save(fig,'04-validation')
