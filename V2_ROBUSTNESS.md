# V2 medically grounded robustness

V2 extends the original noise-aware lesion-preserving graph without using a
generative diffusion model or knowledge distillation. The V1 command-line
defaults remain unchanged; V2 is enabled explicitly by `train_v2.bat` and
`test_v2.bat` and writes to a separate checkpoint directory.

## Training pipeline

For every source-domain sample, V2 draws from five generic intensity
perturbation mechanisms:

- `additive`
- `magnitude`
- `signal_dependent`
- `multiplicative`
- `low_frequency`

These names describe mathematical corruption mechanisms rather than claiming
that every acquisition modality physically contains every noise family. The
V2 fine-tuning command keeps 70% of primary inputs clean and perturbs 30%.
Two additional independent perturbations are generated directly from the
original image and used as uncertainty probes; noise is not stacked on an
already perturbed primary input.

V2 uses the same fixed severity distribution throughout fine-tuning instead of
a three-stage curriculum. `train_v2.bat` samples severity uniformly from
`0.00` to `0.06` in every epoch. Noise types are sampled uniformly unless
`--train_noise_weights` is supplied. This keeps the robustness objective a
small regularizer around the pretrained representation rather than making
late, strong synthetic corruption the dominant task.

Source masks never alter spatial augmentation. They are used only to reduce an
overly strong intensity perturbation when it would retain less than the
configured fraction of local lesion/background contrast. This contrast guard
is not used at inference.

## Feature graph

Auxiliary views estimate patch uncertainty but are not averaged into the graph
input:

```text
primary image -----------> primary patch features ---> soft graph ---> output
       |                            ^
       +--> independent probes ----+ uncertainty only
```

The soft graph combines continuous semantic affinity and spatial adjacency.
Source-node reliability decreases with patch uncertainty, while anomaly
probability suppresses cross-boundary propagation and reduces the receiver
update gate. Spectral normalization constrains the graph projection operator.

V2 keeps the original classification, segmentation, lesion-preservation, and
boundary losses. It averages consistency across independent probe views and
penalizes variance between those view losses, avoiding a hard worst-noise
objective. The V2 command trains the image adapter for five epochs, saves every
epoch, and uses smaller auxiliary-loss weights than V1. A held-out validation
set should select the checkpoint; the final epoch is not assumed to be best.

## Commands

Train Brain V2:

```bat
train_v2.bat
```

Test all V2 image checkpoints on both clean and noisy Liver inputs:

```bat
test_v2.bat
```

The two evaluations use the same checkpoints and produce separate summary
files for severity `0.0` and `0.06`. This exposes both clean accuracy and the
corruption-induced drop instead of judging robustness from noisy accuracy
alone.

The target dataset selects the default medically corresponding corruption:

- Brain: Rician magnitude noise plus a low-frequency bias field
- Liver: signal-dependent CT quantum-noise approximation
- DDTI: multiplicative speckle plus weak electronic noise
- Retina: shot noise plus weak additive noise
- Colon: illumination variation plus weak additive noise

Use `--test_noise_type` or `--probe_noise_type` to override `auto` for an
ablation. Keep `--test_noise_severity 0` for clean-primary evaluation and sweep
`0.02`, `0.04`, `0.06`, `0.08`, and `0.10` for corruption robustness.

## Checkpoint compatibility

V1 and V2 use different graph behavior and parameterization. Store them in
separate directories:

```text
ckpt/noise_graph
ckpt/noise_graph_v2
ckpt/noise_graph_v2_finetune
```

V2 image checkpoints record their architecture configuration. Evaluation stops
with an explicit error when soft-graph, spectral-normalization, or primary-only
flags do not match the checkpoint instead of silently evaluating the wrong
model.

The lightweight fine-tuning revision uses `ckpt/noise_graph_v2_finetune` so
an earlier three-stage V2 checkpoint cannot be resumed accidentally.

The dataset loader and `full-shot.jsonl` selection mechanism are unchanged.
