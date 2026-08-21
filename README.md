# SBI-reasoning toy model

This repository contains a compact demonstration of data-profile-guided conditional neutrino reconstruction. It follows the three stages in `Discussion-Proposal_For_SBI_alignment.pdf`:

1. an observable-only SBI profiler produces the data-preferred parent-mass estimate;
2. a second profiler estimates the mass using observables plus a proposed neutrino reconstruction;
3. Direct Group Preference Optimization (DGPO) fine-tunes a conditional rectified-flow model so the reconstructed and observed profiles agree.

The DGPO adaptation follows [Luo, Hu, and Tang, *Reinforcing Diffusion Models by Direct Group Preference Optimization*](https://arxiv.org/abs/2510.08425) and its [official implementation](https://github.com/Luo-Yihong/DGPO). The toy keeps deterministic ODE rollouts, group-normalized advantages, shared forward-process noise within each group, a frozen reference model, a minimum training timestep, and the group DSM objective. Each pseudo-data mass independently fine-tunes a fresh copy of the same supervised-flow checkpoint with a mass-specific, order-independent random stream. The preference reward uses only the squared difference between the calibrated observable-only and reconstructed-profile best-fit mass estimators. It does not compare likelihood shapes.

## Physics setup

Each event contains two equal-mass parents with back-to-back three-momenta. The parent momentum magnitude is sampled directly from a non-negative two-sided exponential (Laplace) distribution centred at 300 GeV with a 45 GeV scale, and the energy is then fixed by `E = sqrt(p^2 + M^2)`. Each parent decays isotropically in its rest frame to one lepton and one invisible particle. The measured inputs are the two smeared lepton four-vectors and missing transverse momentum. The learned target is the pair of invisible-particle three-momenta; invisible energy is computed from the configured mass shell before reconstructing both parent invariant masses.

The early-trial templates are exactly 400, 500, and 600 GeV. Nominal pseudo-data cover 400--600 GeV in 25 GeV steps. No intermediate simulated mass is added for profiler training. All masses, event counts, detector effects, architecture sizes, training settings, plots, and hypothesis grids live in YAML.

The nominal mass profiler is selected with one YAML field:

```yaml
model:
  profiler_type: mass_parameterized_ratio  # or template_classifier
```

`mass_parameterized_ratio` trains a joint-versus-product ratio estimator with mass as an input, using events from only the three template masses. `evaluation.profile_scan_masses_gev` is an inference-only scan over that trained conditional model; it does not generate additional simulation. `template_classifier` retains the original three-class implementation and ignores the scan grid.

The repository supports three hypothesis experiments:

1. a one-dimensional parent-mass hypothesis test at nominal detector scale;
2. a one-dimensional JES-like scale test at fixed parent mass with -10%, 0%, and +10% templates;
3. a two-dimensional parent-mass x JES template grid.

`config/toy.yaml` deliberately enables only the first, mass-only experiment. It is the early-trial workflow: learn the two neutrino three-momenta, compare the observable-only and visible-plus-invisible mass profiles, apply DGPO on that one-dimensional profile, and validate momentum and reconstructed-mass closure. `config/nersc.yaml` and `config/smoke.yaml` enable all three experiments. The selection is controlled by `experiments.enabled`.

The event contains leptons rather than reconstructed jets, so the JES study is explicitly a detector-response proxy. A coherent visible-object scale shift is applied to both reconstructed visible four-vectors, and the opposite transverse shift is propagated to missing momentum. It is not a replacement for a shower, hadronisation, and jet-calibration simulation.

## Run

```bash
uv sync
uv run sbi-toy --config config/toy.yaml --stage all
```

The stages can also be run separately with `--stage generate`, `--stage train`, and `--stage evaluate`. A fast integration check is available with:

```bash
uv run sbi-toy --config config/smoke.yaml --stage all
uv run python -m unittest discover -s tests -v
```

Outputs are written under the configured `output_dir`:

- `data/`: generated template and pseudo-data samples;
- `dataset_truth_mass_sanity.*`: energy sampling, truth mass closure, oracle spectra, and peak closure;
- `sbi_input_distributions.*`: the ten raw low-level inputs used by both SBI models, overlaid for every parent-mass template with common bins and per-hypothesis density normalization;
- `sbi_input_distributions.csv`: bin counts and densities underlying every low-level input panel;
- `models.pt`: both profilers, their selected type, the three simulated masses, the inference profile grid, the pretrained flow, one independent DGPO flow per pseudo-data mass, early-stop reasons, profile-calibration anchors, and scalers;
- `training_history.*`: discriminator, flow, and DGPO monitoring;
- `dgpo_early_stopping.csv`: per-pseudo-data stop reason, completed iterations, final estimator gap, and final reward spread;
- `momentum_reconstruction_diagnostics.*`: truth-versus-reconstructed neutrino momentum and component-wise residual bias and resolution at the configured diagnostic mass;
- `momentum_reconstruction_samples.csv` and `momentum_reconstruction_metrics.csv`: source data for the momentum diagnostic;
- `sbi_likelihood_diagnostics.*`: visible versus visible-plus-truth-invisible mass profiles;
- `profile_classifier_validation.csv`: held-out raw mass estimates and biases at the three simulated anchors for both profilers;
- `profile_calibration.csv`: held-out raw-to-calibrated profile-mass anchors;
- `profile_quality_metrics.csv`: per-pseudo-data curvature, dynamic range, boundary status, and calibrated bias;
- `jes_likelihood_diagnostics.*`: visible versus visible-plus-truth-invisible JES profiles;
- `mass_jes_likelihood_grid.*`: two-dimensional likelihood-score surfaces;
- `mass_jes_profile_scatter.*`: truth versus profiled mass; hue identifies the method, light-to-dark shade identifies truth JES, and marker size identifies truth mass in the reciprocal JES panel;
- `mass_jes_reconstruction_heatmaps.*`: reconstructed peak bias and Gaussian mass resolution across the full truth mass x JES validation grid, with one row per configured reconstruction method;
- `mass_jes_reconstruction_metrics.csv`: fitted peak, bias, and resolution source data for every method and grid point;
- `reconstructed_mass_spectra.*` and `final_benchmark.*`: final reconstruction benchmarks;
- CSV files beside every diagnostic figure and `summary.json`: source data and aggregate metrics.

Publication figures use a compact Nature-style visual contract and are exported as editable SVG, PDF, 600 dpi TIFF, and preview PNG in the full configurations. The plotted likelihood diagnostic is the relative mean profiler score, labelled as `-2 Delta <log p>`. In parameterized mode the curve is the inference-only mass scan and the markers identify the three simulated templates; in legacy mode the curve interpolates the three classifier outputs. Profile vertices are calibrated only with held-out simulated template samples; raw and calibrated estimates are both retained in the CSV output. Invalid non-monotonic anchors or near-random profiler accuracy stop formal training before DGPO.

DGPO has two independent stopping guards for every pseudo-data mass. It stops after the calibrated full-profile estimator remains within `estimator_gap_tolerance_gev` of the visible estimator for the configured patience, or when `reward_spread` remains below its threshold. The NERSC and formal toy defaults use a 5 GeV estimator gap; `iterations_per_pseudo_data` is only the maximum.

The exact figure contract, statistical definitions, and current validation boundary are recorded in `FIGURE_QA.md`. A single configured seed and an 80/20 train/validation split are used by default; the code does not claim multi-seed confidence intervals. Peak error bars show the fitted event-level mass resolution, not uncertainty on the fitted mean.

## NERSC GPU run and W&B

Prepare the environment on a Perlmutter login node:

```bash
uv sync --frozen
```

Store the W&B key outside the repository in `~/.env` as `WANDB_API_KEY=...`. The loader never prints the key. Then submit one shared-QOS GPU job, supplying the NERSC account at submission time:

```bash
sbatch --account=<NERSC_ACCOUNT> scripts/train_nersc.slurm
```

`config/nersc.yaml` requests CUDA with bfloat16 autocasting, larger GPU batches, online W&B logging, three simulated masses at 300, 500, and 700 GeV, and all mass/JES experiments. The Slurm script follows the Perlmutter one-GPU shared-QOS layout and deliberately leaves the account outside source control.

The reported peak and resolution come from a Gaussian-plus-offset fit to the reconstructed mass histogram. If a fit is ill-conditioned, evaluation falls back to the median and central 68% half-width.

## Scope

This is a method-validation toy, not an identifiable measurement of an unknown invisible-particle rest mass. The default invisible mass is zero and the model reconstructs invisible momentum. A nonzero mass shell can be configured in YAML, but learning that common rest mass from unlabeled dilepton-plus-MET events would require adding it as a global parameter of interest and validating identifiability and coverage.
