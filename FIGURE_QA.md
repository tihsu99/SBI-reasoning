# Figure QA record

## Scientific claim and hierarchy

Core claim: the generator closes at truth level, visible-plus-invisible information sharpens the classifier profile, and DGPO can be benchmarked by mass bias, resolution, and profile alignment under nominal and JES-shifted detector response.

Archetype: quantitative grid. `final_benchmark`, `mass_jes_profile_scatter`, and `mass_jes_reconstruction_heatmaps` are the hero figures; generator, one-dimensional likelihood, JES, and two-dimensional likelihood panels are diagnostics.

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
- Classifier metric: validation classification accuracy.
- Likelihood diagnostic: `-2 Delta <log p>`, the relative mean classifier log-score. Markers are the evaluated templates; curves are quadratic interpolation.
- Reconstructed peak: Gaussian-plus-offset fit; fallback is the median if the fit fails.
- Resolution: fitted Gaussian sigma, or central 68% half-width for the fallback. It is not a confidence interval on the peak.
- Bias: fitted reconstructed peak minus truth parent mass.
- Mass x JES reconstruction heatmaps: each cell is fitted independently from the reconstructed event-level mass spectrum at that truth grid point; these are not classifier-profile bias or profile uncertainty.
- JES scatter error bar: standard deviation of mass bias across evaluated truth masses at fixed JES shift. It is not a standard error.
- Baselines: MET-split neutrino momentum, supervised conditional flow, and detector oracle as labelled.

## Validation status

- Unit tests cover two-body mass closure, truncated double-exponential energy sampling, JES-to-MET propagation, and the DGPO gradient direction.
- The complete smoke configuration has passed all stages and rendered every figure.
- Smoke outputs validate integration and layout only; they are not physics results because the smoke run uses 256 template events and two training epochs.
- Final NERSC numerical results and the resulting figure content remain to be produced with `config/nersc.yaml`.
