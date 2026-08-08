# SBI-reasoning toy model

This repository contains a compact demonstration of data-profile-guided conditional neutrino reconstruction. It follows the three stages in `Discussion-Proposal_For_SBI_alignment.pdf`:

1. an observable-only template discriminator produces the data-preferred parent-mass profile;
2. a second discriminator profiles the mass using observables plus a proposed neutrino reconstruction;
3. Direct Group Preference Optimization (DGPO) fine-tunes a conditional rectified-flow model so the reconstructed and observed profiles agree.

The DGPO adaptation follows [Luo, Hu, and Tang, *Reinforcing Diffusion Models by Direct Group Preference Optimization*](https://arxiv.org/abs/2510.08425) and its [official implementation](https://github.com/Luo-Yihong/DGPO). The toy keeps deterministic ODE rollouts, group-normalized advantages, shared forward-process noise within each group, a frozen reference model, a minimum training timestep, and the group DSM objective.

## Physics setup

Each event contains two equal-mass parents. Their energies follow a two-sided exponential (Laplace) distribution centred at 1,000 GeV and analytically conditioned on `energy >= parent mass`. Their three-momenta are back-to-back. Each parent decays isotropically in its rest frame to one lepton and one invisible particle. The measured inputs are the two smeared lepton four-vectors and missing transverse momentum. The learned target is the pair of invisible-particle three-momenta; invisible energy is computed from the configured mass shell before reconstructing both parent invariant masses.

The default templates are 200, 500, and 800 GeV, with 10,000 events at each point. Nominal pseudo-data cover 200--800 GeV in 25 GeV steps. All masses, event counts, detector effects, architecture sizes, training settings, plots, and hypothesis grids live in YAML.

The repository runs three hypothesis experiments:

1. a one-dimensional parent-mass hypothesis test at nominal detector scale;
2. a one-dimensional JES-like scale test at fixed parent mass with -10%, 0%, and +10% templates;
3. a two-dimensional parent-mass x JES template grid.

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
- `models.pt`: both discriminators, the pretrained flow, the DGPO flow, and scalers;
- `training_history.*`: discriminator, flow, and DGPO monitoring;
- `sbi_likelihood_diagnostics.*`: visible versus visible-plus-truth-invisible mass profiles;
- `jes_likelihood_diagnostics.*`: visible versus visible-plus-truth-invisible JES profiles;
- `mass_jes_likelihood_grid.*`: two-dimensional likelihood-score surfaces;
- `mass_jes_profile_scatter.*`: truth versus profiled mass; hue identifies the method, light-to-dark shade identifies truth JES, and marker size identifies truth mass in the reciprocal JES panel;
- `mass_jes_reconstruction_heatmaps.*`: reconstructed peak bias and Gaussian mass resolution across the full truth mass x JES validation grid, with one row per configured reconstruction method;
- `mass_jes_reconstruction_metrics.csv`: fitted peak, bias, and resolution source data for every method and grid point;
- `reconstructed_mass_spectra.*` and `final_benchmark.*`: final reconstruction benchmarks;
- CSV files beside every diagnostic figure and `summary.json`: source data and aggregate metrics.

Publication figures use a compact Nature-style visual contract and are exported as editable SVG, PDF, 600 dpi TIFF, and preview PNG in the full configurations. The plotted likelihood diagnostic is the classifier-based mean event log-score, labelled as `-2 Delta <log p>`; the three template evaluations are shown explicitly and the continuous curves are quadratic interpolation.

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

`config/nersc.yaml` requests CUDA with bfloat16 autocasting, larger GPU batches, online W&B logging, the full 200--800 GeV pseudo-data range, and the three JES evaluation slices. The Slurm script follows the Perlmutter one-GPU shared-QOS layout and deliberately leaves the account outside source control.

The reported peak and resolution come from a Gaussian-plus-offset fit to the reconstructed mass histogram. If a fit is ill-conditioned, evaluation falls back to the median and central 68% half-width.

## Scope

This is a method-validation toy, not an identifiable measurement of an unknown invisible-particle rest mass. The default invisible mass is zero and the model reconstructs invisible momentum. A nonzero mass shell can be configured in YAML, but learning that common rest mass from unlabeled dilepton-plus-MET events would require adding it as a global parameter of interest and validating identifiability and coverage.
