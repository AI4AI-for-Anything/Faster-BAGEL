"""Four-seed pre-registered statistics; no result-dependent thresholds."""
import math,statistics,warnings
from scipy import stats

def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def mean(xs):return math.fsum(xs)/len(xs) if xs and all(finite(x) for x in xs) else None

def compare(ref,opt):
    out={'reference_seed_values':ref,'candidate_seed_values':opt,'n_reference':len(ref),'n_candidate':len(opt),'defined':len(ref)==len(opt)==4 and all(finite(x) for x in ref+opt)}
    if not out['defined']:
        return dict(out,reference_mean=None,candidate_mean=None,sigma_ref=None,sigma_opt=None,delta=None,D=None,paired_differences=None,paired_t=None,paired_t_p=None,paired_t_df=3,paired_t_note='Undefined input; no omitted seed',consistent=False)
    mr=mean(ref);mo=mean(opt);sr=statistics.stdev(ref);so=statistics.stdev(opt);d=[o-r for o,r in zip(opt,ref)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        t=stats.ttest_rel(opt,ref,alternative='two-sided')
    # Undefined/degenerate inferential results are explicit and are not arm nonfinites.
    note='two-sided paired Student t-test, four seed pairs; descriptive, not a pass check'
    if not finite(float(t.statistic)) or not finite(float(t.pvalue)):note+='; degenerate difference variance, nonfinite statistic represented as null'
    return dict(out,reference_mean=mr,candidate_mean=mo,sigma_ref=sr,sigma_opt=so,delta=2*sr,D=mo-mr,paired_differences=d,paired_t=float(t.statistic) if finite(float(t.statistic)) else None,paired_t_p=float(t.pvalue) if finite(float(t.pvalue)) else None,paired_t_df=3,paired_t_note=note,consistent=abs(mo-mr)<=2*sr)

def training_result(windows):
    assert len(windows)==20
    failed=[w['start_step'] for w in windows if not w['consistent']]
    defined=all(w['defined'] for w in windows)
    positive=sum(w['defined'] and w['D']>0 for w in windows)
    negative=sum(w['defined'] and w['D']<0 for w in windows)
    ties=sum(w['defined'] and w['D']==0 for w in windows)
    n=positive+negative
    p=float(stats.binomtest(positive,n,.5,alternative='two-sided').pvalue) if defined and n else (1.0 if defined else None)
    return {'windows':windows,'failed_windows':len(failed),'failed_window_start_steps':failed,'allowed_failed_windows':2,'final_window_pass':windows[-1]['consistent'],'passes':defined and len(failed)<=2 and windows[-1]['consistent'],'bias':{'positive_D_windows':positive,'negative_D_windows':negative,'zero_D_ties':ties,'nonzero_windows':n,'two_sided_binomial_p':p,'flag_p_lt_0_01':p<.01 if p is not None else None,'rule':'exact two-sided Binomial(n_nonzero, 0.5) sign test; exact-zero D ties reported and excluded; descriptive flag only'}}
