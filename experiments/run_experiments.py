"""CPU experiments. All seeds/configurations are declared here before execution.

Linear experiments sample the exact Gaussian OLS sufficient-statistic law.
The nonlinear-feature experiment fits OLS on actual generated reset data.
No validation outcomes or true parameters select a deployable method.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.geometry import (metric, controller, spectral_compress, regret,
    compression_loss, certificate, gaussian_radii, draw_ols, oracle_bound,
    psd_sqrt, lift_lti, fit_ols)


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def balanced_forecast(dh, bh, rank, state, actions, horizon, c):
    """Finite-horizon balanced-state baseline; same full action map retained.

    Pilot A,B use the first-step rows of the same fitted rollout map.
    This baseline uses the known time layout and observation weighting C.
    """
    a = np.linalg.solve(c, dh[:state])
    b = np.linalg.solve(c, bh[:state, :actions])
    wc = np.zeros((state, state)); wo = wc.copy(); ak = np.eye(state)
    for _ in range(horizon):
        wc += ak @ b @ b.T @ ak.T
        wo += ak.T @ c.T @ c @ ak
        ak = ak @ a
    rc, ro = psd_sqrt(wc), psd_sqrt(wo)
    u, s, vt = np.linalg.svd(ro.T @ rc)
    # An explicit numerical floor handles essentially uncontrollable modes.
    invroot = np.diag(1 / np.sqrt(np.maximum(s[:rank], 1e-12)))
    v = rc @ vt[:rank].T @ invroot
    e = invroot @ u[:, :rank].T @ ro.T
    ar = e @ a @ v
    return np.vstack([c @ v @ np.linalg.matrix_power(ar, k) @ e
                      for k in range(1, horizon + 1)])


def evaluate_fit(d, b, dh, bh, gram, sigma, ranks, base, balanced=None):
    rows=[]
    rad = gaussian_radii(d.shape[0], d.shape[1], b.shape[1], gram, sigma)
    wh, wt = metric(bh), metric(b)
    event = float(np.linalg.norm(dh-d, 'fro') <= rad.forecast_fro and
                  np.linalg.norm(bh-b, 2) <= rad.action_op)
    for rank in ranks:
        fo, po = spectral_compress(d, wt, rank)
        oracle = regret(d,b,fo,b)
        for method in ['prediction','plugin','robust','oracle_geometry','full']:
            w = {'prediction':np.eye(d.shape[0]), 'plugin':wh,
                 'robust':wh+rad.eta*np.eye(d.shape[0]),
                 'oracle_geometry':wt, 'full':np.eye(d.shape[0])}[method]
            r = d.shape[1] if method=='full' else rank
            f, proj = spectral_compress(dh,w,r)
            loss=regret(d,b,f,bh)
            rows.append(dict(base, rank=rank, method=method, regret=loss,
                oracle=oracle, forecast_mse=float(np.linalg.norm(f-d,'fro')**2),
                geometry_excess=max(0.,compression_loss(d,b,proj)-
                                    (0. if method=='full' else oracle)),
                certificate=certificate(dh,bh,f,rad), eta=rad.eta,
                event=event, oracle_bound=oracle_bound(d,b,dh,rank,rad)
                if method=='robust' else float('nan')))
        if balanced is not None:
            f=balanced(dh,bh,rank)
            rows.append(dict(base,rank=rank,method='balanced_state',
                regret=regret(d,b,f,bh),oracle=oracle,
                forecast_mse=float(np.linalg.norm(f-d,'fro')**2),
                geometry_excess=float('nan'),certificate=certificate(dh,bh,f,rad),
                eta=rad.eta,event=event,oracle_bound=float('nan')))
    return rows


def linear_suite(reps):
    rows=[]; rng=np.random.default_rng(20260915)
    ql,_=np.linalg.qr(rng.normal(size=(6,6)))
    qr,_=np.linalg.qr(rng.normal(size=(6,6)))
    qu,_=np.linalg.qr(rng.normal(size=(6,6)))
    ds=np.array([3.,2.2,1.5,1.,.6,.3])
    for scenario, bs in [('aligned_weak',[.08,.075,.06,.05,.03,.02]),
                         ('conflicting',[.03,.06,.12,.4,1.,2.])]:
        d=ql@np.diag(ds)@qr.T; b=ql@np.diag(bs)@qu.T
        for coverage in [.03,1.]:
            for n in [48,192,768,3072]:
                for seed in range(reps):
                    rng=np.random.default_rng(100000+seed+100*n+int(coverage*100))
                    dh,bh,gram=draw_ols(d,b,n,.2,coverage,rng)
                    base=dict(suite='linear',scenario=scenario,coverage=coverage,
                              n=n,transitions=n,seed=seed)
                    rows.extend(evaluate_fit(d,b,dh,bh,gram,.2,[1,2,3],base))
    return rows


def horizon_suite(reps):
    rows=[]
    a=np.array([[.82,.7,0,0],[0,.65,0,0],[0,0,.8,.4],[0,0,0,.7]])
    b=np.array([[0.,0],[1,0],[0,0],[0,1]])
    c=np.diag([1.,np.sqrt(.1),1.,np.sqrt(.1)])
    horizon=6; d,b=lift_lti(a,b,horizon,c)
    for n in [64,256,1024,4096]:
        for seed in range(reps):
            rng=np.random.default_rng(200000+100*n+seed)
            dh,bh,gram=draw_ols(d,b,n,.03,.3,rng)
            base=dict(suite='horizon',scenario='nonnormal_H6',coverage=.3,
                      n=n,transitions=n*horizon,seed=seed)
            balanced=lambda dh,bh,r: balanced_forecast(dh,bh,r,4,2,horizon,c)
            rows.extend(evaluate_fit(d,b,dh,bh,gram,.03,[1,2,3],base,balanced))
    return rows


def features(x):
    return np.vstack([x,np.sin(x),x[0]*x[1],x[2]*x[3]])


def nonlinear_suite(reps):
    rows=[]; rng=np.random.default_rng(31)
    cov=np.eye(10); cov[4:8,4:8]*=(1-np.exp(-2))/2
    cov[:4,4:8]=np.eye(4)*np.exp(-.5); cov[4:8,:4]=cov[:4,4:8].T
    root=psd_sqrt(cov); invroot=np.linalg.inv(root)
    coefficients=rng.normal(size=(4,10))*np.r_[np.ones(4),np.ones(4)*.7,[.5,.5]]
    d=coefficients@root
    b=np.diag([.06,.2,.7,1.2]); sigma=.1
    for coverage in [.1,1.]:
        for n in [64,256,1024]:
            for seed in range(reps):
                rng=np.random.default_rng(300000+n*100+seed+int(coverage*10))
                x=rng.normal(size=(4,n)); h=invroot@features(x)
                u=rng.normal(size=(4,n))*np.sqrt(coverage)
                z=np.vstack([h,u]); y=d@h+b@u+rng.normal(size=(4,n))*sigma
                estimate=fit_ols(z,y)
                base=dict(suite='nonlinear_features',scenario='sine_product_dictionary',
                          coverage=coverage,n=n,transitions=n,seed=seed)
                rows.extend(evaluate_fit(d,b,estimate[:,:10],estimate[:,10:],
                                         z@z.T,sigma,[1,2,3],base))
    # A separate moment check does not train or select any model.
    rng=np.random.default_rng(777); h=invroot@features(rng.normal(size=(4,200000)))
    return rows, float(np.linalg.norm(h@h.T/h.shape[1]-np.eye(10),2))


def rate_suite(reps):
    rows=[]; sigma=.5; d=np.eye(2)
    for n in [32,128,512,2048,8192,32768]:
        for regime in ['local_tie','fixed_gap']:
            delta=sigma/(4*np.sqrt(2*n)) if regime=='local_tie' else .04
            b=np.diag([1+delta,1-delta]); wt=metric(b)
            gap=wt[0,0]-wt[1,1]
            for seed in range(reps):
                rng=np.random.default_rng(400000+n*10+seed)
                bh=b+rng.normal(size=(2,2))*sigma/np.sqrt(n)
                _,p=spectral_compress(d,metric(bh),1)
                excess=max(0.,compression_loss(d,b,p)-wt[1,1])
                rows.append(dict(n=n,regime=regime,seed=seed,delta=delta,
                    gap=gap,excess=excess,scaled_excess=excess*np.sqrt(n)/sigma,
                    lower_bound=3*gap/8 if regime=='local_tie' else float('nan')))
    return rows


def local_expansion():
    rng=np.random.default_rng(42)
    d=rng.normal(size=(4,3)); b=rng.normal(size=(4,2))
    ed=rng.normal(size=d.shape); eb=rng.normal(size=b.shape)
    k,h=controller(b)
    dk=np.linalg.solve(h,eb.T-(eb.T@b+b.T@eb)@k)
    du=dk@d+k@ed; quadratic=float(np.sum(du*(h@du)))
    rows=[]
    for t in np.logspace(-1,-4,13):
        exact=regret(d,b,d+t*ed,b+t*eb)
        rows.append(dict(t=t,exact=exact,quadratic=t*t*quadratic,
                         remainder=abs(exact-t*t*quadratic)))
    return rows


def rank_selection(reps):
    rows=[]
    d=np.diag([3.,2.2,1.5,1.,.6,.3]); b=np.diag([.3,.2,.15,.1,.06,.03])
    for n in [48,192,768,3072,12288]:
        for seed in range(reps):
            rng=np.random.default_rng(500000+n+seed)
            dh,bh,gram=draw_ols(d,b,n,.02,1.,rng)
            rad=gaussian_radii(6,6,6,gram,.02)
            candidates=[]
            for r in range(7):
                f,_=spectral_compress(dh,metric(bh)+rad.eta*np.eye(6),r)
                candidates.append((r,certificate(dh,bh,f,rad),regret(d,b,f,bh)))
            for tolerance in [.02,.1,.3]:
                valid=[x for x in candidates if x[1]<=tolerance]
                chosen=valid[0] if valid else (-1,float('nan'),float('nan'))
                rows.append(dict(n=n,seed=seed,tolerance=tolerance,
                                 rank=chosen[0],certificate=chosen[1],regret=chosen[2]))
    return rows


def summarize(rows, group, fields):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in group); groups.setdefault(key,[]).append(row)
    out=[]
    for key,values in groups.items():
        record=dict(zip(group,key)); record['replicates']=len(values)
        for field in fields:
            arr=np.array([v[field] for v in values]); arr=arr[np.isfinite(arr)]
            record[field+'_mean']=float(arr.mean()) if len(arr) else float('nan')
            record[field+'_se']=float(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr)>1 else 0.
        out.append(record)
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--reps',type=int,default=40)
    parser.add_argument('--rate-reps',type=int,default=1000)
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    rows=linear_suite(args.reps)+horizon_suite(args.reps)
    nonlinear,moment_error=nonlinear_suite(args.reps); rows+=nonlinear
    rate=rate_suite(args.rate_reps); expansion=local_expansion(); selected=rank_selection(args.reps)
    write_csv(args.out/'raw.csv',rows); write_csv(args.out/'rate_raw.csv',rate)
    write_csv(args.out/'local_expansion.csv',expansion); write_csv(args.out/'rank_selection_raw.csv',selected)
    groups=['suite','scenario','coverage','n','rank','method']
    write_csv(args.out/'summary.csv',summarize(rows,groups,
        ['regret','oracle','forecast_mse','geometry_excess','certificate','eta','event']))
    write_csv(args.out/'rate_summary.csv',summarize(rate,['n','regime'],
        ['excess','scaled_excess','lower_bound']))
    write_csv(args.out/'rank_selection_summary.csv',summarize(selected,['n','tolerance'],['rank','regret','certificate']))
    slope=float(np.polyfit(np.log([r['t'] for r in expansion][4:10]),
                           np.log([r['remainder'] for r in expansion][4:10]),1)[0])
    robust=[r for r in rows if r['method']=='robust']
    certified=[r for r in selected if r['rank']>=0]
    metadata=dict(date='2026-09-15',python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,reps=args.reps,rate_reps=args.rate_reps,
        blas_threads=os.environ.get('OPENBLAS_NUM_THREADS','not recorded'),
        elapsed_seconds=time.perf_counter()-start,raw_rows=len(rows),rate_rows=len(rate),
        local_remainder_slope=slope,nonlinear_feature_covariance_check_op_error=moment_error,
        confidence_event_failures=sum(r['event']==0 for r in robust),
        certificate_violations=sum(r['regret']>r['certificate']+1e-9 for r in rows),
        oracle_bound_violations=sum(r['regret']>r['oracle_bound']+1e-9 for r in robust),
        rank_selections=len(selected),certified_rank_selections=len(certified),
        selected_tolerance_violations=sum(r['regret']>r['tolerance']+1e-9 for r in certified),
        selection_abstentions=len(selected)-len(certified),
        notes=['Linear and horizon data use exact Gaussian OLS sufficient statistics.',
               'Horizon n counts reset rollouts; transitions=n*6.',
               'Nonlinear dictionary is fixed and analytically whitened, not learned.',
               'Oracle geometry uses true B only as a diagnostic.',
               'Counts include correlated multiple-rank uses of the same training draw.',
               'Intervals in summary are Monte Carlo standard errors, not simultaneous CIs.'])
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
