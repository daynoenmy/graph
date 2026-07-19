# Self-supervised diffusion denoising

> Legacy experiment: the current AA-CLIP training and evaluation path uses
> feature-space noise uncertainty and does not load diffusion or student
> denoiser checkpoints. The standalone stages below are retained only for
> reproducing earlier denoising experiments.

This repository implements the following offline training pipeline:

```text
noisy training images
  -> masked blind-spot network
  -> conditional diffusion teacher
  -> symmetric DDIM pseudo-clean images
  -> lightweight deterministic input denoiser
  -> frozen CLIP + trainable AA-CLIP adapters
```

The blind-spot, diffusion, and student stages use only `image_path` from the
metadata. Class labels and segmentation masks are not read. Use a metadata file
containing training images only; do not include the test set.

## DDTI data

The commands below use Windows CMD syntax. Set paths for your machine:

```bat
set DDTI_DATA_PATH=D:\datasets\DDTI
set DATA_ROOT=%DDTI_DATA_PATH%
set META_PATH=dataset\metadata\DDTI\full-shot.jsonl
set DENOISE_DIR=ckpt\ddti_denoising
```

Each metadata line must at least contain a path relative to `DATA_ROOT`:

```json
{"image_path": "images/example.png"}
```

DDTI ultrasound images are trained as one-channel images. The distilled module
converts CLIP's RGB input to grayscale, denoises it, repeats it to three
channels, and then applies CLIP normalization.

## 1. Train the masked blind-spot network

```bat
python train_denoiser.py train-blind-spot ^
  --data-root "%DATA_ROOT%" ^
  --metadata-path "%META_PATH%" ^
  --image-size 256 ^
  --channels 1 ^
  --batch-size 8 ^
  --epochs 50 ^
  --mask-probability 0.05 ^
  --output-dir "%DENOISE_DIR%\blind_spot"
```

Resume a stopped run with:

```bat
  --resume "%DENOISE_DIR%\blind_spot\blind_spot_latest.pth"
```

## 2. Cache blind-spot condition images

Full J-invariant reconstruction requires multiple network passes, so its output
is cached once before diffusion training:

```bat
python train_denoiser.py cache-conditions ^
  --data-root "%DATA_ROOT%" ^
  --metadata-path "%META_PATH%" ^
  --batch-size 4 ^
  --blind-spot-checkpoint "%DENOISE_DIR%\blind_spot\blind_spot_latest.pth" ^
  --blind-spot-stride 4 ^
  --output-dir "%DENOISE_DIR%\conditions"
```

## 3. Train the conditional diffusion teacher

The observed DDTI image is the diffusion training sample and the cached
blind-spot reconstruction is the condition:

```bat
python train_denoiser.py train-diffusion ^
  --data-root "%DATA_ROOT%" ^
  --metadata-path "%META_PATH%" ^
  --condition-root "%DENOISE_DIR%\conditions" ^
  --image-size 256 ^
  --channels 1 ^
  --batch-size 4 ^
  --epochs 200 ^
  --timesteps 1000 ^
  --amp ^
  --output-dir "%DENOISE_DIR%\diffusion"
```

Use a smaller `--batch-size` if GPU memory is insufficient. Resume with
`--resume ...\diffusion_latest.pth`.

## 4. Generate pseudo-clean targets

By default this performs two DDIM trajectories initialized with `z` and `-z`
and averages their outputs:

```bat
python train_denoiser.py generate-pseudo-clean ^
  --data-root "%DATA_ROOT%" ^
  --metadata-path "%META_PATH%" ^
  --condition-root "%DENOISE_DIR%\conditions" ^
  --diffusion-checkpoint "%DENOISE_DIR%\diffusion\diffusion_latest.pth" ^
  --sampling-steps 50 ^
  --batch-size 1 ^
  --output-dir "%DENOISE_DIR%\pseudo_clean"
```

## 5. Distill the lightweight input denoiser

```bat
python train_denoiser.py train-student ^
  --data-root "%DATA_ROOT%" ^
  --metadata-path "%META_PATH%" ^
  --pseudo-clean-root "%DENOISE_DIR%\pseudo_clean" ^
  --image-size 256 ^
  --channels 1 ^
  --batch-size 8 ^
  --epochs 30 ^
  --width 32 ^
  --depth 5 ^
  --output-dir "%DENOISE_DIR%\student"
```

The resulting checkpoint is:

```text
ckpt\ddti_denoising\student\input_denoiser.pth
```

## Legacy denoising ablations

Evaluate downstream anomaly detection and localization, not only visual image
quality:

1. AA-CLIP baseline without input denoising.
2. Blind-spot condition image as direct CLIP input.
3. Distilled deterministic student output.

Inspect difference images around lesion boundaries to ensure that denoising did
not remove diagnostically relevant structures.
