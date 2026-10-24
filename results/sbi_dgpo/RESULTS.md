# SBI-guided DGPO with the moment-curve estimator

**Code:**
- `sbi_reasoning/moment_curve.py`: the estimator.
- `train_dgpo(..., estimator=...)`: an optional hook. Without it, DGPO behaves exactly as before.
- `scripts/sbi_dgpo.py`: one full comparison per seed.
- `scripts/sbi_dgpo_figures.py`: tables and figures.

All numbers below come from 4 seeds of `scripts/sbi_dgpo.py` at its defaults (toy.yaml physics,
100k events per template and per dataset, the pipeline's training and DGPO settings).

## What is compared

Each truth mass (400–600 GeV in 25 GeV steps) has two independent, unlabeled 100k-event datasets.
Post-training adapts on the first; every metric is measured on the second.

| method | what it is | uses dataset-level information? |
|---|---|---|
| `sft` | the supervised flow trained on the 400/500/600 templates | no |
| `truth_dgpo` | DGPO with a per-event truth reward on template simulation (NeurIPS-style); one model for all masses | no |
| `sbi_toy` | DGPO per dataset, reward −((m̂_full(reco) − m̂_visible(data)) / 100 GeV)², toy estimators (the current pipeline) | yes |
| `sbi_new` | the same reward with the moment-curve estimators | yes |
| `sbi_new_shifted` | `sbi_new` aimed at m̂_visible − δ, where δ is the peak offset `sbi_new` produces on fresh template simulation (known mass) | yes |
| `cond_sft` | a flow conditioned on the parent mass, trained on the templates and run at the visible-only moment-curve estimate. No DGPO; the proposal's "template-conditioned" baseline | yes |
| `oracle` | true neutrinos | – |

**Only simulation truth is used.** Truth enters training and calibration only through simulation:
the templates, and the fresh template datasets used to measure δ. The target datasets are used
only through their visible quantities, and `train_dgpo` never reads their neutrinos or mass.

## Metric

**Reconstructed-mass bias** (primary):
1. Add each lepton to its reconstructed neutrino to get a parent mass, for both parents of every
   event.
2. Fit the peak of that mass histogram (the pipeline's `fit_mass_peak`).
3. Bias = peak − true mass. The fitted width is the resolution.
4. Slope and nonlinearity come from a straight-line fit of peak vs truth. A reconstruction that
   follows the data has slope 1.

**Secondary metrics:**
- **Per-event accuracy:** RMSE of the neutrino momentum components against truth (the NeurIPS
  paper's metric).
- **W1:** the Wasserstein-1 distance to the true-neutrino mass spectrum.
- **Drift:** the RMS of the mean signed neutrino residual.

## Results (4 seeds, averaged over the 9 truth masses)

| method | peak bias (signed) | \|peak bias\| | slope | width | ν RMSE | W1 | drift |
|---|---|---|---|---|---|---|---|
| supervised flow | +1.8 | 30.2 ± 0.6 | 0.46 | 73.4 | 288.6 | 72.4 | 5.7 |
| truth-reward DGPO | −14.5 | 33.5 ± 0.8 | 0.43 | 63.6 | **266.9** | 66.9 | **4.1** |
| SBI-DGPO, toy estimators | +20.1 | 20.6 ± 4.2 | 1.06 (S-shaped, nonlinearity 12.9) | 72.3 | 295.8 | 69.4 | 22.9 |
| SBI-DGPO, moment curve | +18.4 | 18.4 ± 1.2 | 1.00 | 71.6 | 294.3 | 68.1 | 19.7 |
| SBI-DGPO, moment curve, shifted target | +2.8 | 2.8 ± 0.2 | 0.99 | 68.4 | 287.3 | 65.7 | 18.6 |
| **mass-conditioned flow at the SBI estimate** | **+1.0** | **1.9 ± 1.2** | **0.99** | **16.9** | 273.7 | **21.4** | 8.3 |
| true neutrinos | −0.1 | 0.5 | 1.00 | 7.5 | 0 | 0 | 0 |

All values are in GeV except slope; ± is the run-to-run sd of the mass-averaged |bias|. The
visible-only moment-curve estimate itself is within 0.9 GeV of the truth at every mass.

Figures:
- `fig_spectra.png`: reconstructed mass spectra at 425/500/575 GeV, and the neutrino pz residual.
- `fig_peak.png`: peak and its bias vs truth.
- `fig_metrics.png`: ν RMSE, W1, drift and the full-event estimate vs truth.

### Reading the results

1. **Per-event reconstruction cannot know the dataset's mass.** One event identifies its template
   only 57% of the time. A flow trained on an equal mix of 400/500/600 therefore hedges toward
   the middle: peak slope 0.46. That matches what an ideal per-event posterior gives (the
   classifier's mean per-event posterior mass is 464/499/535 for 400/500/600). Truth-reward DGPO
   sharpens each event toward that prior and compresses further (slope 0.43), while giving the
   best per-event RMSE.
2. **SBI guidance fixes the mass dependence.** With the moment-curve estimators the reconstruction
   follows the data exactly (slope 1.00, nonlinearity 0.9 GeV), but with a constant +18 GeV peak
   offset. The toy estimators transmit their S-shaped bias, swinging from +3 to +41 GeV with mass.
3. **Where the offset comes from.** DGPO matches the full-event statistic, and the moment-curve
   statistic is effectively the geometric mean of the reconstructed parent masses, which lands on
   the truth (+0.7 GeV). The reconstructed spectrum is ~70 GeV wide and left-skewed, so its
   median (+15) and peak (+18) sit higher. The shape barely changes with mass, so the offset is
   constant. Shifting the target by the offset measured on template simulation brings the peak
   bias to +2.8 GeV.
4. **The mass-conditioned flow does better than every DGPO variant here, without DGPO.** Given the
   visible SBI estimate as an input, it puts the peak within 1–2 GeV with a 17 GeV width and the
   best W1. This is close to built in: a flow conditioned on m produces neutrinos whose parent mass
   is m, so its peak mostly reproduces the visible estimate, which is already accurate to under
   1 GeV.
   - **What it means for this toy:** the parameter being measured is also the only thing that
     differs between simulation and data. Conditioning the reconstructor on the SBI estimate is
     therefore enough, and policy-gradient adaptation adds nothing.
   - **Where SBI-DGPO would be needed:** when the reconstructor cannot simply be conditioned. For
     example, a pretrained model such as EveNet that is not retrained, or a nuisance (JES-like)
     that shifts between simulation and data while a different quantity is measured.

## Masses outside the templates (2 seeds, same method)

| truth (GeV) | 350 | 375 | 400 | 500 | 600 | 625 | 650 |
|---|---|---|---|---|---|---|---|
| visible moment-curve estimate − truth | +1.8 | +1.2 | +0.9 | −0.2 | +0.7 | +1.0 | −0.0 |
| SBI-DGPO shifted: peak − truth | +6.6 | +4.9 | +3.6 | +2.3 | +2.0 | +0.5 | −2.9 |
| supervised flow: peak − truth | +89 | +72 | +56 | −1 | −53 | −67 | −80 |

Up to 50 GeV beyond the templates the method degrades smoothly. It is limited by construction to
the search window: templates ± 50 GeV for the visible estimate. At 650 the estimate sits on the
window edge, and masses beyond it are clipped. To cover a wider range, use templates that bracket it.

## Caveats

- Truth-reward DGPO is scaled down from the paper's: 4 candidates, 1000 iterations, no EMA rollout
  and no anchor term. It improved per-event RMSE as expected, but it is not tuned.
- The SBI-DGPO variants roughly triple the mean-residual drift (5.7 → 19). That is the
  distribution shift the NeurIPS paper's mean-residual anchor targets, and the two are untested in
  combination.
- The estimator design (feature set, search window) and the offset shift were developed on this
  400–600 grid with toy.yaml physics. The visible estimator alone was also checked on a held-out
  physics configuration (parent momentum 300 ± 45 GeV); the full SBI-DGPO chain was not.

## Reproduce

```bash
for seed in 0 1 2 3; do uv run python scripts/sbi_dgpo.py --seed $seed; done   # ~12 min each on a T4
uv run python scripts/sbi_dgpo_figures.py outputs/sbi_dgpo results/sbi_dgpo
```
