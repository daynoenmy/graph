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
that every acquisition modality physically contains every noise family. By
default, 70% of primary inputs are perturbed and 30% stay clean. Two additional
independent perturbations are used as uncertainty probes.

The maximum severity follows a three-stage curriculum. With a final maximum of
`0.10`, the stages use maxima of `0.03`, `0.06`, and `0.10` for the first 25%,
next 35%, and final 40% of image-adapter epochs. Noise types are sampled
uniformly unless `--train_noise_weights` is supplied.

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
objective.

## Commands

Train Brain V2:

```bat
train_v2.bat
```

Test all V2 image checkpoints on noisy Liver inputs:

```bat
test_v2.bat
```

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
```

V2 image checkpoints record their architecture configuration. Evaluation stops
with an explicit error when soft-graph, spectral-normalization, or primary-only
flags do not match the checkpoint instead of silently evaluating the wrong
model.

The dataset loader and `full-shot.jsonl` selection mechanism are unchanged.
