# AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP
 **[CVPR 2025 paper]**

[![Paper](https://img.shields.io/badge/CVPR-Paper-red)](https://arxiv.org/pdf/2503.06661) [![Appendix](https://img.shields.io/badge/CVPR-Appendix-blue)](https://drive.google.com/file/d/1PQrjCvWDyuM7W2ClJ-cJeD4YKJ1uPAzc/view?usp=drive_link)
 Official Pytorch Implementation

![](pic/teaser.png)

## Abstract
Anomaly detection (AD) identifies outliers for applications like defect and lesion detection. While CLIP shows promise for zero-shot AD tasks due to its strong generalization capabilities, its inherent **Anomaly-Unawareness** leads to limited discrimination between normal and abnormal features. To address this problem, we propose **Anomaly-Aware CLIP** (AA-CLIP), which enhances CLIP's anomaly discrimination ability in both text and visual spaces while preserving its generalization capability. AA-CLIP is achieved through a straightforward yet effective two-stage approach: it first creates anomaly-aware text anchors to differentiate normal and abnormal semantics clearly, then aligns patch-level visual features with these anchors for precise anomaly localization. This two-stage strategy, with the help of residual adapters, gradually adapts CLIP in a controlled manner, achieving effective AD while maintaining CLIP's class knowledge. Extensive experiments validate AA-CLIP as a resource-efficient solution for zero-shot AD tasks, achieving state-of-the-art results in industrial and medical applications. 

## Results
![](pic/results.png)

## Quick Start 
### 1. Installation  
```bash
git clone https://github.com/Mwxinnn/AA-CLIP.git
cd AA-CLIP
conda create -n aaclip python=3.10 -y  
conda activate aaclip  
pip install -r requirements.txt  
```
### 2. Datasets
The datasets can be downloaded from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BrainMRI, LiverCT, Retinafrom](https://drive.google.com/drive/folders/1La5H_3tqWioPmGN04DM1vdl3rbcBez62?usp=sharing) from [BMAD](https://github.com/DorisBao/BMAD), [CVC-ColonDB, CVC-ClinicDB, Kvasir, CVC-300](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) from Polyp Dataset.

Put all the datasets under ``./data`` and use jsonl files in ``./dataset/metadata/``. You can use your own dataset and generate personalized jsonl files with below format:
```json
{"image_path": "xxxx/xxxx/xxx.png", 
 "label": 1.0 (for anomaly) # or 0.0 (for normal), 
 "class_name": "xxxx", 
 "mask_path": "xxxx/xxxx/xxx.png"}
```
The way of creating corresponding jsonl file differs, depending on the file structure of the original dataset. The basic logic is recording the path of every image file, mask file, anomaly label ``label``, and category name, then putting them together under a jsonl file.

(Optional) the base data directory can be edited in ``./dataset/constants``. If you want to reproduce the results with few-shot training, you can generate corresponding jsonl files and put them in ``./dataset/metadata/{$dataset}`` with ``{$shot}-shot.jsonl`` as the file name. For few-shot training, we use ``$shot`` samples from each category to train the model.

> Notice: Since the anomaly scenarios in VisA are closer to real situations, the default hyper-parameters are set according to the results trained on VisA. More analysis and discussion will be available.

### 3. Training & Evaluation
```bash
# training
python train.py --shot $shot --save_path $save_path
# evaluation
python test.py --save_path $save_path --dataset $dataset
# (Optional) we provide bash script for training and evaluating all the datasets
bash scripts.sh
```

### 4. Noise-aware lesion-preserving patch graph

The current training path uses two mask-aligned, modality-aware intensity views
to estimate patch uncertainty. The patch graph then propagates reliable normal
features while suppressing smoothing around likely lesions and their boundaries.
It does not reconstruct images and requires neither a diffusion model nor a
distilled denoiser. See ``train.bat`` for Brain source training and ``test.bat``
for a Brain-to-Liver cross-dataset example.

The image branch also retains CLIP's final CLS output as a global semantic
anchor. By default, the image representation uses a 0.2 CLS residual, and the
medical image score combines 0.8 pixel-map maximum with 0.2 global score:

```bash
python train.py --dataset Brain --training_mode full_shot \
  --save_path ./ckpt/noise_graph_cls_llm \
  --prompt_source llm --llm_prompt_path ./dataset/llm_prompts.json \
  --clip_global_weight 0.2 --global_text_temperature 10.0

python test.py --dataset Liver --save_path ./ckpt/noise_graph_cls_llm \
  --image_checkpoint 'image_adapter_*.pth' \
  --prompt_source llm --llm_prompt_path ./dataset/llm_prompts.json \
  --clip_global_weight 0.2 --global_text_temperature 10.0 \
  --medical_image_score_global_weight 0.2
```

The normal/abnormal anchors are ensembles of modality-specific visual
descriptions authored offline with an LLM and stored in
``dataset/llm_prompts.json``. No LLM API is called during training or testing.
The prompt source and prompt-bank SHA256 are stored in every new checkpoint;
loading a checkpoint with a different bank is rejected. The global score is
the abnormal probability from a temperature-scaled normal/abnormal Softmax,
not a rescaled abnormal cosine alone.

Set both fusion weights to zero for the pre-CLS V1 ablation. Use
``--prompt_source template`` when evaluating a checkpoint trained before the
LLM prompt bank was introduced. The batch files use a separate
``noise_graph_cls_llm`` directory, so previous V1 checkpoints are not
overwritten. ``test.bat`` evaluates every numbered image checkpoint by default.

The former diffusion-denoising experiment is retained for reference in
[DENOISING.md](DENOISING.md), but its checkpoints are not consumed by the
current ``train.py`` or ``test.py`` pipeline.

### 5. V3.1: frozen lesion-preserving spatial-frequency graph

The independent V3 path freezes the complete CLIP image and text encoders and
trains only a small post-encoder head. It applies a fixed stationary Haar
transform independently to frozen layer 6/12/18/24 patch grids, applies one
shared local spatial-frequency coherence graph head, and fuses the four layer
probabilities with learned non-negative weights. Neighbor consensus distinguishes
supported lesions from isolated pseudo-anomalies. Its bounded residual head
corrects the fixed normal/abnormal CLIP margin without unrestricted semantic
drift. Image scoring combines focal Top-k evidence, diffuse GeM evidence, and
the frozen final CLIP CLS global evidence; the CLS path does not alter the pixel
map. V3 does not use V1 adapters, a second noisy image view, KNN graph, or the
V1 boundary-contrast loss.

```bash
python train_v3.py --dataset Brain --training_mode full_shot \
  --feature_layers 6 12 18 24 \
  --save_path ./ckpt/v3_multilayer_sfgraph --prompt_source llm

python test_v3.py --dataset Liver \
  --feature_layers 6 12 18 24 \
  --save_path ./ckpt/v3_multilayer_sfgraph \
  --checkpoint 'v3_head_epoch_*.pth' --test_noise_severity 0.0
```

See [V3_METHOD.md](V3_METHOD.md) for the equations, architecture differences,
medical interpretation, commands, and required ablations. The original V1
entry points remain ``train.py`` and ``test.py``.

BMAD mixed-supervision leave-one-dataset-out is supported with explicit mask
availability and dataset-balanced source sampling:

```bat
train_v3_lodo.bat Chest
test_v3_lodo.bat Chest
```

See [BMAD_LODO.md](BMAD_LODO.md) for dataset paths, optional-mask JSONL schema,
the six folds, loss routing, metric rules, and target-domain leakage controls.

### 6. V4.2-SSC: semantic-spectral coupling

V4.2-SSC keeps frozen CLIP layers
6/12/18/24 and constructs fixed four-neighbor Laplacian bases `margin`,
`L margin`, and `L^2 margin`. The original CLIP margin remains the semantic
backbone. A small conditioner predicts four non-negative layer weights and
bounded signed `L/L^2` residual coefficients instead of treating all twelve
bases as interchangeable evidence. The conditioner uses a separate fixed
hand-written modality phrase such as `a brain MRI scan` or `a chest X-ray`.
Three frozen focal/diffuse/structural text prototypes are compared with the
normal anchor using independent sigmoid compatibility scores. A shared zero-
initialized `3x2` matrix adds global image-specific semantic corrections to the
bounded `L/L^2` coefficient logits. No online LLM is called.

The final CLS/local readout is fixed. V4.2-SSC trains only 9,234 parameters for
ViT-L/14 and uses Image BCE plus valid-mask Pixel Focal/Dice; it does not use
the V3 Haar, graph head, lesion gate, band intervention, Top-k, or GeM modules.
Both training batch files expose `--pixel_loss_weight`, so the Pixel Loss
contribution can be changed directly without editing Python.

```bat
train_v4_lodo.bat Chest
test_v4_lodo.bat Chest
```

For the `Chest` fold, these two scripts apply the same conservative X-ray
preset: Laplacian temperature `0.15`, uniform layer mass `0.35`, maximum
spectral coefficient `0.75`, aspect temperature `7.5`, readout temperature
`1.5`, and source-domain Pixel Loss weight `0.5`. The preset uses only the
known target modality/mask-availability protocol, not Chest images or test
metrics. Report it as a target-specific preset rather than a single shared-
hyperparameter six-fold LODO result.

See [V4_2_TEXT_SPECTRAL_COUPLING.md](V4_2_TEXT_SPECTRAL_COUPLING.md) for the
semantic compatibility, coupling equation, mixed-supervision routing, and
checkpoint contract. V4.2-SSC checkpoints are stored under `ckpt/v4_2_ssc` or
`ckpt/v4_2_ssc_bmad_lodo/<TARGET>` and cannot be mixed with V4.1, V3, or older
checkpoints.

For a paired medical case study against the original AA-CLIP, run:

```bash
python case_analysis.py \
  --dataset Liver \
  --aa_save_path ./ckpt/baseline \
  --aa_image_checkpoint image_adapter_1.pth \
  --v1_save_path ./ckpt/noise_graph_cls_llm \
  --v1_image_checkpoint image_adapter_1.pth \
  --output_dir ./case_results/Liver
```

The script exports per-image statistics, representative median cases, selected
case metrics, and AA-CLIP/V1/uncertainty comparison panels. Select both
checkpoints without using the target test set.

To explain how V1 treats four representative patch states in one real case,
run:

```bash
python visualize_patch_states.py \
  --dataset Liver \
  --save_path ./ckpt/noise_graph_cls_llm \
  --image_checkpoint image_adapter_1.pth \
  --label anomaly \
  --case_index 0 \
  --noise_severity 0.06 \
  --output_dir ./patch_state_examples/Liver
```

The script visualizes uncertainty, anomaly probability, clean/noisy prediction
disagreement, source reliability, the learned graph gate, four post-hoc patch
states, and exact graph source weights. The four colors are explanatory
thresholds only; V1 itself uses continuous scores rather than a four-class
head.

Model definition is in ``./model/``. We thank [```open_clip```](https://github.com/mlfoundations/open_clip.git) for being open-source. To run the code, one has to download the weight of OpenCLIP ViT-L-14-336px and put it under ```./model/```.

## Additional Discussion
(I am writing down my experimental observations and thoughts. In this part, it is less formal and rigorous.)
We have observed several interesting phenomenons during our experiments:

### The impact of class-level supervision differs across domains.
In the initial stage of adapting the text encoder, we tried to apply binary cross-entropy (BCE) loss to directly distinguish between normal and anomalous text embeddings—an approach that imposes stronger supervision than our current method, which adds class embeddings to visual embeddings and relies on segmentation loss for separation. Experimental results indicate that BCE improves zero-shot performance on industrial datasets but negatively affects performance in the medical domain. This may be due to the lower diversity and simpler structure of anomaly representations in medical data, making them easier to learn without strong supervision.

### Adaptation hyper-parameters should be carefully tuned.
 Since CLIP is pre-trained on a massive dataset and anomaly detection is a comparatively simpler task, the model is prone to issues like catastrophic forgetting or overfitting. Careful control of the adaptation process is essential, which is why our method involves multiple hyper-parameters (though these can still be further optimized).

Additionally, the differences in anomaly patterns between the training dataset and downstream zero-shot datasets cannot be ignored. Overfitting to the anomaly characteristics of the training set can lead the model to rely on superficial cues. For instance, if the training data predominantly features round-shaped anomalies, the model may prioritize shape over true semantic understanding of anomalies. Incorporating training data with a wider variety of anomaly types could help mitigate this issue.

### To be updated...

## Citation
If you use this work, please cite:
```
@misc{ma2025aaclipenhancingzeroshotanomaly,
      title={AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP}, 
      author={Wenxin Ma and Xu Zhang and Qingsong Yao and Fenghe Tang and Chenxu Wu and Yingtai Li and Rui Yan and Zihang Jiang and S. Kevin Zhou},
      year={2025},
      eprint={2503.06661},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06661}, 
}
```

## Contact
For questions or collaborations:

- Email: mwxisj@gmail.com

- GitHub Issues: [Open Issue](https://github.com/Mwxinnn/AA-CLIP/issues)

⭐ Star this repo if you find it useful!
