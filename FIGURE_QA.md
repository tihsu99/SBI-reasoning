# Figure QA record

## Scientific claim and hierarchy

Core claim: the generator closes at truth level, visible-plus-invisible information sharpens the SBI profile, and DGPO can be benchmarked by mass bias, resolution, and profile alignment under nominal and JES-shifted detector response.

The low-level input grid separately checks that the mass hypotheses induce visible shape changes in exactly the ten raw variables supplied to the SBI models. All hypotheses use common bin edges and are normalized to probability density; the accompanying CSV retains counts and densities for every bin.

Early-trial generator claim: direct parent-momentum sampling produces the configured mass-independent double-exponential spectrum while preserving the parent mass shell and two-body truth closure for the 400, 500, and 600 GeV templates.

Archetype: quantitative grid. For the mass-only early trial, `momentum_reconstruction_diagnostics` and `final_benchmark` are the hero figures. For the complete systematics workflow, `mass_jes_profile_scatter` and `mass_jes_reconstruction_heatmaps` are additional hero figures; generator and likelihood panels are diagnostics.

## Nature-style checks

- Final width: 7.2 inches (182.9 mm, double-column target).
- Maximum figure height: below 170 mm.
- Body, tick, and legend text: 6.5--7 pt.
- Panel labels: lowercase, bold, 8 pt.
- Typeface: Arial with DejaVu Sans and Liberation Sans fallbacks.
- Axes: units in parentheses; ticks and axis lines retained; background gridlines omitted.
- Colour: colour-blind-aware blue, orange, and warm red identify methods; light-to-dark shades encode truth JES. Marker shape repeats the method encoding, and marker size identifies truth mass in the reciprocal JES panel.
- Heatmap scales: mass bias uses one zero-centred diverging scale shared by all methods; resolution uses one sequential scale shared by all methods. Method names are direct row labels and both colour bars include GeV units.
- Vector text: `svg.fonttype = none`; PDF TrueType font type 42.
- Export: editable SVG and PDF; full configurations also write 600 dpi TIFF and 300 dpi PNG previews.
- Source data: every quantitative diagnostic has a CSV file in the same output directory.

## Statistical and machine-learning definitions

- `n`: configured events per mass or grid point; displayed on the relevant figure.
- Train/validation split: stratified by template class using the configured validation fraction (default 80/20).
- Randomness: one configured random seed per run; no multi-seed or fold uncertainty is claimed.
- Profiler metric: validation binary accuracy for the joint-versus-product parameterized ratio estimator, or multiclass accuracy for the legacy template classifier.
- Simulation boundary: profiler training always uses exactly the three masses in `data.template_masses_gev`. A dense parameterized curve is an inference scan, not additional simulation.
- Likelihood diagnostic: `-2 Delta <log p>`, the relative mean profiler score. Markers identify the three simulated templates. Parameterized curves are direct inference scans; legacy curves interpolate the three classifier outputs.
- Profile validation: the accuracy panel includes the mode-appropriate chance baseline. Held-out anchor estimates, calibration anchors, score dynamic range, local curvature, and boundary saturation are exported as CSV.
- Profile calibration: observable and full vertices use separate monotonic mappings derived only from held-out simulated templates. Raw vertices remain in source data, and pseudo-data truth is never used for calibration or DGPO reward.
- DGPO independence: every nominal pseudo-data mass starts from the same frozen supervised-flow checkpoint and produces a separate model. The reward is the negative squared difference between the calibrated observable-only and reconstructed-profile best-fit mass estimators; likelihood-shape agreement is not used.
- DGPO stopping: every mass has independent estimator-gap and low-reward-spread patience counters. The configured iteration count is a maximum, not a requirement to continue after the reward becomes uninformative.
- Reconstructed peak: Gaussian-plus-offset fit; fallback is the median if the fit fails.
- Resolution: fitted Gaussian sigma, or central 68% half-width for the fallback. It is not a confidence interval on the peak.
- Bias: fitted reconstructed peak minus truth parent mass.
- Momentum diagnostic: scatter panels show the same configured sample of neutrino entries for each method; the summary panel reports mean residual bias with one standard deviation, not uncertainty on the mean.
- Mass x JES reconstruction heatmaps: each cell is fitted independently from the reconstructed event-level mass spectrum at that truth grid point; these are not classifier-profile bias or profile uncertainty.
- JES scatter error bar: standard deviation of mass bias across evaluated truth masses at fixed JES shift. It is not a standard error.
- Baselines: MET-split neutrino momentum, supervised conditional flow, and detector oracle as labelled.

## Validation status

- Unit tests cover two-body mass closure, truncated double-exponential energy and momentum sampling, JES-to-MET propagation, experiment-mode validation, the three-mass profiler switch, dense ratio-scan evaluation, local profile estimation, and the DGPO gradient direction.
- The complete smoke configuration has passed all stages and rendered every figure.
- A separate mass-only smoke integration has passed generation, training, evaluation, and momentum-figure rendering without creating JES or mass x JES datasets.
- Smoke outputs validate integration and layout only; they are not physics results because the smoke run uses 256 template events and two training epochs.
- Final NERSC numerical results and the resulting figure content remain to be produced with `config/nersc.yaml`.
