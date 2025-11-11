# Time-Series-Koopman

This repository contains a PyTorch implementation of the **LKH-SDE v3** architecture described in the accompanying research note. The project now provides:

- A stability-constrained Koopman mixture-of-experts module with horizon-adaptive decay, convex expert combination, and Van-Loan discretisation for covariance propagation.
- A multi-scale CI-HiTS head that fuses coarse-to-fine temporal convolutions with optional known-future priors.
- A gated fusion layer with curriculum-aware scheduling, back-half weighting, and covariance scaling for strict probabilistic decoding when the Koopman readout uses `tkernel=1`.
- Training utilities that implement the two-stage curriculum, frequency-domain regularisation, Koopman semigroup/stitching penalties, entropy control, optional EMA, and diagnostics logging.
- A configurable data pipeline with reversible instance normalisation, seasonal feature generation, and support for known-future covariates.

## Quick start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a CSV dataset containing a timestamp column followed by target variables (see `data/sample.csv` for a synthetic example). Update `configs/example.yaml` with the desired paths and hyper-parameters.

3. Launch training:

   ```bash
   python train.py --config configs/example.yaml
   ```

The script warms up the CI-HiTS branch for `stage1_epochs` epochs before unlocking the Koopman branch. Stage two increases the Koopman gate multiplier over `stage2_warmup_epochs` while annealing the expert temperature. Checkpoints, EMA snapshots, and diagnostic logs are written to the directories specified in the configuration.

## Configuration highlights

- **Dataset section** – controls window length, horizon, stride, and optional known-future feature columns. Time features (hour, day-of-week, etc.) are generated automatically.
- **Model section** – exposes Koopman segmentation, decay target/scale, expert rank, CI-HiTS dilations and reductions, and fusion hyper-parameters including the gate bias.
- **Optim section** – governs the curriculum (`stage1_epochs`, `stage2_warmup_epochs`), frequency-domain regularisation, Koopman regularisers (semigroup, stitching, entropy), probabilistic losses (`nll_weight`, `crps_weight`), EMA, diagnostics, and checkpointing.

See `configs/example.yaml` for a full template covering long-horizon training with diagnostics enabled.

## Project layout

```
lkh_sde/
  data/            # Dataset utilities and windowing logic
  models/          # LKH-SDE v3 model and configuration dataclasses
  modules/         # Encoder, CI-HiTS, Koopman MoE, fusion building blocks
  trainer.py       # Training loop with curriculum, regularisers, and logging
configs/           # YAML experiment configurations
train.py           # Command-line entry point
```

## Experiment tips

- Set `decay_target` and `decay_scale` to keep the Koopman branch from saturating (values in `[0.6, 0.8]` with scale `0.6–1.0` match the paper).
- Increase `semigroup_weight` / `stitch_weight` when long horizons destabilise or when segment transitions become jagged.
- Enable probabilistic training by keeping the Koopman readout at `tkernel=1`, turning on `nll_weight`/`crps_weight`, and monitoring the logged diagnostics to tune temperature scaling.
- Use the JSONL diagnostics (see `diagnostics_path`) to inspect $\|A_s\|_F$, noise budgets, gate activations, and spectral radii per segment.

With these components you can reproduce the ablations and long-horizon baselines discussed in the research text, extend them with additional priors, or plug the modules into custom pipelines.
