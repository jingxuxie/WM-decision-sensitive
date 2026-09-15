# Executed results — September 15, 2026

The main suite used 40 training draws per condition. Entries below are exact
expected excess planning costs, averaged over training data, with one standard
error across those draws. They are not errors estimated from a finite number
of deployment trajectories. Model choices share the same pilot fit per seed.

| Setting | n | Rank | Prediction | Plug-in geometry | Inflated geometry | Full (different capacity) |
|---|---:|---:|---:|---:|---:|---:|
| Weak aligned actuation, coverage .03 | 192 | 1 | .3628 ± .0265 | .4258 ± .0239 | .3704 ± .0266 | .6210 ± .0289 |
| Conflicting geometry, coverage .03 | 768 | 1 | .5019 ± .0063 | .3290 ± .0085 | .5024 ± .0063 | .1292 ± .0080 |
| Six-step dynamics, coverage .3 | 256 rollouts | 2 | .1319 ± .0001 | .0320 ± <.00005 | .0322 ± <.00005 | .0004 ± <.00005 |
| Fixed nonlinear dictionary, coverage .1 | 256 | 2 | 1.4063 ± .0172 | .1520 ± .0007 | .1880 ± .0015 | .0123 ± .0010 |

Full is not a capacity-matched baseline. In the horizon experiment, n counts
six-step reset rollouts, not single transitions. Oracle geometry is a privileged
diagnostic available in the complete generated summaries. Balanced-state
forecasts are tested only in the horizon suite; their construction retains the
same full action map as the other methods.

Inflation does **not** uniformly improve empirical control. The conflicting
case above is deliberately retained. The proof includes an exact example with
zero plug-in regret and inflation regret 1/2.

## Statistical rate experiment

The forecast is D=I2 and the true action map is revealed only after selecting the
rank-one representation. This isolates geometry selection from deployment
calibration. Each row averages 1,000 independently sampled calibration fits.

| Probes n | Near-tie excess | Fixed-gap excess |
|---:|---:|---:|
| 32 | .0064421 | .0120347 |
| 128 | .0032062 | .0067446 |
| 512 | .0016806 | .0016910 |
| 2048 | .0008506 | .0004178 |
| 8192 | .0004236 | .0000985 |
| 32768 | .0002182 | .0000249 |

Full precision and standard errors are in `rate_summary.csv`. The near-tie
instance changes with n as specified by the minimax construction. Its scaled
excess sqrt(n)/sigma stays between .0725 and .0791. This is not a claim of a
slow rate on every fixed, separated system. Scalar inflation leaves the
isotropic-D eigenspace unchanged, so this experiment does not compare inflated
and plug-in algorithms.

## Calibration and rank certification

There are 1,040 main training draws, 16,080 correlated method/rank evaluations,
12,000 rate-experiment rows, and 600 tolerance queries on 200 additional fits.
No main certificate or oracle-inequality violation was observed. The rank test
certified 560 requests, abstained on 40, and had zero observed tolerance violations
among certified choices. Multiple ranks/tolerances on one fit are not independent
confidence trials. Zero observed violations do not establish tightness or allow
relaxing the statistical assumptions.

At tolerance .1, the smallest certified ranks for n=48,192,768,3072,12288 were
4,3,3,2,2. The strictest tolerance .02 caused abstention at n=48.

## Execution and reproducibility

The recorded complete experimental run took 5.618 seconds using one BLAS thread
in the available CPU environment, excluding plotting, tests, and compilation.
It does not benchmark a particular consumer laptop. Nine numerical unit/property
tests passed. A second complete run reproduced all seven generated experiment
CSVs byte-for-byte in the same environment; see `replication_check.json`.

All raw seed-level data and the complete 402-row main aggregate summary are in
the conversation package. They are regenerated in Git by:

```
python experiments/run_experiments.py
python experiments/make_figures.py
```

The above table is rounded for readability. The generated CSVs retain full
precision. Synthetic experiments, written proofs, and numerical tests are not
independent expert review, formal verification, or evidence of real-world
robot performance.
