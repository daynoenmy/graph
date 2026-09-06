# HSPG-CLIP

## Branch
use origin/hspg

## Results
![](https://github.com/daynoenmy/graph/blob/hspg/pic/line_charts_3x2.pdf)
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



### 3. Training & Evaluation
```bash
# training
python train.py --shot $shot --save_path $save_path
# evaluation
python test.py --save_path $save_path --dataset $dataset
# (Optional) we provide bash script for training and evaluating all the datasets
bash scripts.sh
```

### Leave-one-dataset-out training

The training entry point can hold out one medical dataset and train the existing
AA-CLIP adapters on the other five. The CLIP model, adapter placement, four
feature levels (`6, 12, 18, 24`), and optimizer parameter groups are unchanged.

Place each metadata file at
`dataset/metadata/<dataset>/full-shot.jsonl` and set every dataset root directly
in `dataset/constants.py`.

```bash
python train.py \
  --leave_out Chest \
  --datasets Chest Liver Brain OCT2017 RESC HIS \
  --training_mode full_shot \
  --save_path ckpt/leave_one_out/leave_out_Chest

python test.py \
  --dataset Chest \
  --save_path ckpt/leave_one_out/leave_out_Chest
```

Repeat with each of `Chest`, `Liver`, `Brain`, `OCT2017`, `RESC`, and `HIS` as
`--leave_out`, using a separate checkpoint directory for every fold.

`Chest`, `OCT2017`, and `HIS` are treated as maskless datasets by default. When
they are training sources, they contribute only to the image-level
classification loss; they are excluded from text-stage and patch-level
segmentation losses. When held out for testing, image AUC/AP use the
`det_feature` classification score and pixel AUC/AP are `NaN`.
Override the default when necessary with `--maskless_datasets`.

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
