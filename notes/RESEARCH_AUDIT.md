# Research audit — September 15, 2026

## Status and priorities

This is a complete, executable research draft with written proofs and actually
run synthetic experiments, not an independently reviewed or submitted paper.
No acceptance or final novelty claim is made. The argument narrowed from a
proposed arbitrary reduced latent dynamical model to **finite-horizon forecast
compression with an uncompressed learned action map**. This restriction must
remain visible in the abstract, introduction, theorem scope, and presentation.

## Claim ledger

1. Exact quadratic regret and known-metric rank truncation: background derivation,
   not claimed as novel. Close precedent: Cheng et al., NeurIPS 2021,
   *Data Sharing and Compression for Cooperative Networked Control*.
2. Task-dependent parameter sensitivity: prior work includes Wagenmaker,
   Simchowitz, and Jamieson, ICML 2021, *Task-Optimal Exploration in Linear
   Dynamical Systems*. The paper does not invent task geometry.
3. State/sensitivity model reduction: Otto, Padovan, and Rowley, CoBRAS,
   *Model Reduction for Nonlinear Systems by Balanced Truncation of State and
   Gradient Covariance*. The paper does not invent sensitivity-aware reduction.
4. Spectral gap-free versus fixed-gap excess rates: classical background,
   explicitly credited to Reiss and Wahl, *Nonasymptotic Upper Bounds for the
   Reconstruction Error of PCA*. The paper does not claim the rate dichotomy
   itself is new.
5. Developed here: a block-resolvent global actuation perturbation bound, a
   simultaneous all-representations decision certificate, a spectral oracle
   inequality with forecast/action calibration radii, and a control-realizable
   energy lower bound for a frozen encoder. Their precise novelty still needs
   an independent search and expert comparison.
6. The scalar inflation has a minimax interpretation only for an OUTER metric
   ambiguity set. That set includes physically unrealizable metrics. The
   spectral algorithm minimizes only the residual part of the certificate.
7. A deterministic example proves that inflation can hurt even when the pilot
   equals the truth. A confidence certificate is not an empirical-dominance
   guarantee or a deployment safety guarantee beyond its assumptions.

## Proof checks and delicate assumptions

- Costs have no factor 1/2: the exact regret identity uses H=I+B.T@B directly.
- Feature second moment is known and whitened; arbitrary unit changes without
  transforming the weights are not claimed invariant.
- Unknown actuation affects both importance weights and the deployed controller;
  the certificate includes both effects instead of treating B as known.
- A single deterministic confidence event controls every F. Rank selection is
  therefore adaptive on that same dataset without a union bound over ranks.
  Repeated future data collection is not automatically time-uniform.
- OLS radii assume independent Gaussian output noise with known scale and a
  full-rank fixed/reset design. On-trajectory dependent designs, unknown noise,
  and missing excitation are not silently included.
- Spectral oracle inequality has no eigengap assumption; its dimensions, tail
  energy, and coverage dependence are explicit. It does not prove a globally
  optimal solution to an arbitrary neural compression problem.
- The lower bound concerns a rank-one LINEAR encoder frozen before the true
  action map is revealed. Gaussian conditional expectation justifies allowing
  any controller of that encoding. The hard instance approaches a spectral tie
  as calibration energy grows; it is not a fixed-gap lower bound.
- The claimed 1/sqrt(energy) lower bound permits adaptive probes but imposes a
  pathwise total action-energy budget. KL is evaluated with the chain rule.
- Written mathematical proofs are not equivalent to independent human review
  or formal verification. Nine initial randomized/unit tests passed.

## Experimental accounting and negative findings

Main tests use 1,040 training draws; 16,080 method/rank rows are NOT independent
training datasets. Rate tests add 12,000 draws. Rank selection uses another
200 fits and 600 correlated tolerance queries. Standard errors are across
training draws and are not simultaneous confidence intervals.

Linear and horizon suites use exact Gaussian OLS sufficient-statistic draws.
The horizon design has independent stacked measurement noise, not accumulated
process noise, and n denotes six-step reset rollouts rather than transitions.
The fixed nonlinear dictionary is analytically whitened; it is neither a
learned encoder nor a nonlinear-action planner. A separate state-only moment
check is diagnostic and not used for training or model selection.

The same pilot fit is shared across methods. Full is not capacity-matched;
oracle geometry is privileged. Balanced-state forecasts use the same pilot
and retain the common full action map, so this is not a comprehensive benchmark
against every controller-reduction method.

Inflation helps some weak-aligned cases and hurts conflicting cases. At rank 1,
n=768 and weak action coverage in the conflicting system, plug-in mean regret
is about .329 and inflated regret .502. The method is not advertised as the
best-performing model. Rank certificates are conservative: 40 strict-tolerance
queries abstain despite allowing full rank. Zero numerical violations are
reported honestly, without implying bound tightness or universal validation.

## Concrete pre-submission work

Independent review should prioritize the simultaneous certificate and energy
lower-bound proof, novelty against task-aware compression/model reduction, and
whether the forecast-only rank restriction is sufficiently compelling. A useful
next technical improvement is a tighter physically realizable uncertainty set
or direct minimization of the entire certificate. These are not completed
contributions and must not appear as achieved results.

For broader empirical scope, add a learned feature model and closed-loop
validation only with clearly separated claims; current fixed-feature checks
do not establish either. Restore the untouched official venue style, validate
anonymization, inspect references, and update the AI disclosure after human
review. The user-owned public repository must not be linked in the blinded PDF.
