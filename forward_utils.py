import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import (
    CLASSIFICATION_PROMPTS,
    CLASSIFICATION_REAL_NAMES,
    CLASS_NAMES,
    PROMPTS,
    REAL_NAMES,
)
from model.tokenizer import tokenize
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim

# ================================================================================================
# The following code is used to get criterion for training


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings
prompt_normal = PROMPTS["prompt_normal"]
prompt_abnormal = PROMPTS["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = PROMPTS["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device, adapt_text=True):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence, adapt_text=adapt_text)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device, adapt_text=True):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence, adapt_text=adapt_text)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device, adapt_text=True):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device, adapt_text=adapt_text
        )
        ret_dict[class_name] = text_features
    return ret_dict


def get_classification_single_class_text_embedding(
    model,
    dataset_name,
    class_name,
    device,
):
    """Build global classification anchors with the frozen CLIP text encoder."""
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: "
            f"{CLASS_NAMES[dataset_name]}"
        )
        real_name = CLASSIFICATION_REAL_NAMES.get(dataset_name, {}).get(
            class_name,
            REAL_NAMES[dataset_name][class_name],
        )

    prompt_state = [
        CLASSIFICATION_PROMPTS["prompt_normal"],
        CLASSIFICATION_PROMPTS["prompt_abnormal"],
    ]
    prompt_templates = CLASSIFICATION_PROMPTS["prompt_templates"]
    text_features = []
    for states in prompt_state:
        prompted_states = [state.format(real_name) for state in states]
        prompted_sentences = [
            template.format(state)
            for state in prompted_states
            for template in prompt_templates
        ]
        tokenized_sentences = tokenize(prompted_sentences).to(device)
        class_embeddings = model.encode_text(
            tokenized_sentences,
            adapt_text=False,
        )
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embedding = F.normalize(class_embeddings.mean(dim=0), dim=0)
        text_features.append(class_embedding)
    return torch.stack(text_features, dim=1).to(device)


def get_classification_text_embedding(model, dataset_name, device):
    return {
        class_name: get_classification_single_class_text_embedding(
            model,
            dataset_name,
            class_name,
            device,
        )
        for class_name in CLASS_NAMES[dataset_name]
    }


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    """Create a localization map, averaging maps rather than level features.

    A 4-D input has shape [B, num_levels, num_patches, embedding_dim]. Each
    level is independently compared with the text anchors and converted into
    the same training probability map or test anomaly map. Equal-weight score
    fusion avoids cancelling small lesions in the embedding space.
    """
    if patch_features.ndim == 4:
        if patch_features.shape[1] == 0:
            raise ValueError("patch_features must contain at least one level")
        level_predictions = [
            calculate_similarity_map(
                patch_features[:, level],
                epoch_text_feature,
                img_size,
                test=test,
                domain=domain,
            )
            for level in range(patch_features.shape[1])
        ]
        return torch.stack(level_predictions, dim=0).mean(dim=0)
    if patch_features.ndim != 3:
        raise ValueError(
            "patch_features must have shape [B, L, D] or [B, levels, L, D]"
        )
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


def calculate_image_logits(
    image_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 100.0,
):
    """Calculate normal/abnormal logits from an independent CLS branch."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [B, D]")
    features = F.normalize(image_features, dim=-1)
    if text_embeddings.ndim == 2:
        text = F.normalize(text_embeddings, dim=0)
        logits = features @ text
    elif text_embeddings.ndim == 3:
        if text_embeddings.shape[0] != image_features.shape[0]:
            raise ValueError("batched text embeddings must match image batch size")
        text = F.normalize(text_embeddings, dim=1)
        logits = torch.matmul(features.unsqueeze(1), text).squeeze(1)
    else:
        raise ValueError("text embeddings must have shape [D, 2] or [B, D, 2]")
    if logits.shape[-1] != 2:
        raise ValueError("text embeddings must contain normal and abnormal anchors")
    return logits * temperature


def calculate_image_probability(
    image_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 100.0,
):
    """Calculate image anomaly probability from an independent CLS branch."""
    logits = calculate_image_logits(
        image_features,
        text_embeddings,
        temperature=temperature,
    )
    return torch.softmax(logits, dim=-1)[:, 1]


def calculate_patch_image_probability(
    patch_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 1.0,
    aggregation: str = "mean",
    topk_ratio: float = 0.01,
    quantile: float = 0.99,
):
    """Aggregate patch-level anomaly probabilities into an image score.

    ``mean`` preserves the training-time behavior. ``topk`` and ``quantile``
    are useful at test time when a small lesion would otherwise be diluted by
    the many normal patches in a medical image.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if patch_features.ndim == 4:
        if patch_features.shape[1] == 0:
            raise ValueError("patch_features must contain at least one level")
        level_scores = [
            calculate_patch_image_probability(
                patch_features[:, level],
                text_embeddings,
                temperature=temperature,
                aggregation=aggregation,
                topk_ratio=topk_ratio,
                quantile=quantile,
            )
            for level in range(patch_features.shape[1])
        ]
        return torch.stack(level_scores, dim=0).mean(dim=0)
    if patch_features.ndim != 3:
        raise ValueError(
            "patch_features must have shape [B, L, D] or [B, levels, L, D]"
        )
    features = F.normalize(patch_features, dim=-1)
    if text_embeddings.ndim == 2:
        text = F.normalize(text_embeddings, dim=0)
        logits = features @ text
    elif text_embeddings.ndim == 3:
        if text_embeddings.shape[0] != patch_features.shape[0]:
            raise ValueError("batched text embeddings must match image batch size")
        text = F.normalize(text_embeddings, dim=1)
        logits = torch.matmul(features, text)
    else:
        raise ValueError("text embeddings must have shape [D, 2] or [B, D, 2]")
    if logits.shape[-1] != 2:
        raise ValueError("text embeddings must contain normal and abnormal anchors")
    patch_scores = torch.softmax(logits * temperature, dim=-1)[..., 1]
    if aggregation == "mean":
        return patch_scores.mean(dim=1)
    if aggregation == "max":
        return patch_scores.max(dim=1).values
    if aggregation == "topk":
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        num_topk = max(1, int(np.ceil(patch_scores.shape[1] * topk_ratio)))
        return patch_scores.topk(num_topk, dim=1).values.mean(dim=1)
    if aggregation == "quantile":
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        return torch.quantile(patch_scores.float(), quantile, dim=1)
    raise ValueError(
        "aggregation must be one of: mean, topk, max, quantile"
    )


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


# ================================================================================================


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
    pixel_validity: np.ndarray = None,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )

    # Image-level metrics use the det_feature classification score directly.
    # ================================================================================================
    # pixel level auc & ap (only where pixel ground truth is available)
    if pixel_validity is not None:
        pixel_validity = np.asarray(pixel_validity, dtype=bool)
        pixel_label = pixel_label[pixel_validity]
        pixel_preds = pixel_preds[pixel_validity]
    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()

    if pixel_label.size == 0 or pixel_label.max() == pixel_label.min():
        zero_pixel_auc = np.nan
        zero_pixel_ap = np.nan
    else:
        zero_pixel_auc = roc_auc_score(pixel_label, pixel_preds)
        zero_pixel_ap = average_precision_score(pixel_label, pixel_preds)
    # ================================================================================================
    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
    }
    return result


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    file_names: list[str],
    save_dir: str,
    dataset_name: str,
    class_name: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
        pixel_preds = (pixel_preds * 255).astype(np.uint8)
    if pixel_label.dtype != np.uint8:
        pixel_label = pixel_label != 0
        pixel_label = (pixel_label * 255).astype(np.uint8)
    # ===============================================================================================
    # save path
    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for idx, file in enumerate(file_names):
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, pixel_label.shape[-2:])
        save_image_list = [image]

        if dataset_name == "MVTec":
            damage_name, image_name = file.split("/")[-2:]
            file_name = f"{damage_name}_{image_name}"
        else:
            raise NotImplementedError

        save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
        save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
        save_image_list = save_image_list[:1] + [
            apply_ad_scoremap(image, _) for _ in save_image_list[1:]
        ]
        scoremap = np.vstack(save_image_list)
        cv2.imwrite(os.path.join(save_dir, file_name), scoremap)
