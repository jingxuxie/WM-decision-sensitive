"""Generate publication figures/tables from recorded CSVs; no fitting or tuning."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper'/'figures'
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    with open(ROOT/'results'/name, newline='') as f:
        return list(csv.DictReader(f))

def save(name):
    plt.tight_layout()
    plt.savefig(OUT/(name+'.pdf'), bbox_inches='tight')
    plt.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close()

def main():
    summary=load('summary.csv'); rates=load('rate_summary.csv')
    plt.figure(figsize=(5.6,3.3))
    for regime,label in [('local_tie','Near-tie family'),('fixed_gap','Fixed gap')]:
        rows=[r for r in rates if r['regime']==regime]
        x=np.array([int(r['n']) for r in rows]); y=np.array([float(r['excess_mean']) for r in rows])
        se=np.array([float(r['excess_se']) for r in rows])
        plt.errorbar(x,y,yerr=2*se,marker='o',capsize=3,label=label)
    rows=[r for r in rates if r['regime']=='local_tie']
    plt.plot([int(r['n']) for r in rows],[float(r['lower_bound_mean']) for r in rows], '--',label='Near-tie minimax lower bound')
    plt.xscale('log');plt.yscale('log');plt.xlabel('Calibration probes n');plt.ylabel('Excess over rank-one oracle')
    plt.legend(fontsize=8);save('rates')
    for scenario,title in [('aligned_weak','Weak actuation, aligned prediction geometry'),('conflicting','Conflicting prediction and decision geometry')]:
        plt.figure(figsize=(5.6,3.3))
        for method,label in [('prediction','Prediction'),('plugin','Plug-in geometry'),('robust','Inflated geometry'),('oracle_geometry','Oracle geometry (diagnostic)')]:
            rows=[r for r in summary if r['suite']=='linear' and r['scenario']==scenario and float(r['coverage'])==.03 and int(r['rank'])==1 and r['method']==method]
            plt.errorbar([int(r['n']) for r in rows],[float(r['regret_mean']) for r in rows],yerr=[2*float(r['regret_se']) for r in rows],marker='o',capsize=3,label=label)
        plt.xscale('log');plt.yscale('log');plt.xlabel('Reset transitions n');plt.ylabel('True excess planning cost')
        plt.title(title,fontsize=10);plt.legend(fontsize=8);save(scenario)
    plt.figure(figsize=(5.6,3.2))
    for tol in [.02,.1,.3]:
        rows=[r for r in load('rank_selection_summary.csv') if float(r['tolerance'])==tol]
        plt.plot([int(r['n']) for r in rows],[float(r['rank_mean']) for r in rows],marker='o',label=f'Tolerance {tol:g}')
    plt.xscale('log');plt.yticks([-1,0,1,2,3,4,5],['Abstain','0','1','2','3','4','5'])
    plt.xlabel('Calibration transitions n');plt.ylabel('Smallest certified rank');plt.legend(fontsize=8);save('certified_rank')
    # A compact generated copy contains every main configuration.
    fields=['suite','scenario','coverage','n','rank','method','replicates','regret_mean','regret_se','forecast_mse_mean','certificate_mean']
    with open(ROOT/'results'/'publication_summary.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in summary)
    configurations=[('linear','aligned_weak',.03,192,1,'Aligned, weak coverage'),('linear','conflicting',.03,768,1,'Conflicting, weak coverage'),('horizon','nonnormal_H6',.3,256,2,'Six-step dynamics, rank 2'),('nonlinear_features','sine_product_dictionary',.1,256,2,'Nonlinear dictionary, rank 2')]
    methods=['prediction','plugin','robust','full']
    text=['\\begin{tabular}{lrrrr}','\\toprule','Setting & Prediction & Plug-in & Inflated & Full \\\\','\\midrule']
    for suite,scenario,cov,n,rank,label in configurations:
        cells=[]
        for method in methods:
            r=next(r for r in summary if r['suite']==suite and r['scenario']==scenario and float(r['coverage'])==cov and int(r['n'])==n and int(r['rank'])==rank and r['method']==method)
            cells.append('$%.4f_{\\pm %.4f}$'%(float(r['regret_mean']),float(r['regret_se'])))
        text.append(label+' & '+' & '.join(cells)+' \\\\')
    text+=['\\bottomrule','\\end{tabular}']
    (ROOT/'paper'/'results_table.tex').write_text('\n'.join(text)+'\n')
    print('Saved four figures, a results table, and publication_summary.csv.')
if __name__=='__main__': main()
