import os
import argparse
from collections import Counter
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from utils import (
    GENERIC_MEDICAL_NOISE_TYPES,
    make_medical_noise_view,
    make_random_medical_noise_view,
    preserve_lesion_contrast,
    setup_seed,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
    calculate_noise_consistency_loss,
    calculate_lesion_preservation_losses,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def curriculum_severity_max(epoch, total_epochs, final_max):
    """Three-stage medical noise curriculum ending at ``final_max``."""
    progress = (epoch + 1) / max(total_epochs, 1)
    if progress <= 0.25:
        return final_max * 0.3
    if progress <= 0.60:
        return final_max * 0.6
    return final_max


def train_text_adapter(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    # scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
            # forward text
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2
            # calculate similarity and get prediction
            loss = 0.0
            for f in patch_features:
                # bs,patch_num,768
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2
            loss += orthogonal_loss * text_norm_weight
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
    return adapted_model


def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    dataset_name: str,
    noise_severity: float,
    noise_consistency_weight: float,
    noise_balance_weight: float,
    lesion_preservation_weight: float,
    boundary_contrast_weight: float,
    boundary_margin: float,
    train_noise_types,
    train_noise_weights,
    primary_noise_probability: float,
    noise_severity_min: float,
    noise_severity_max: float,
    num_noise_views: int,
    min_lesion_contrast_retention: float,
    run_config,
    logger: logging.Logger,
):
    # Keep the frozen CLIP encoder in evaluation mode while allowing spectral
    # normalization in the V2 graph projection to update its power iteration.
    model.image_adapter.train()
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        consistency_loss_list = []
        balance_loss_list = []
        preservation_loss_list = []
        boundary_loss_list = []
        primary_noise_counts = Counter()
        reference_noise_counts = Counter()
        sampled_severities = []
        v2_noise_training = train_noise_types is not None
        current_severity_max = curriculum_severity_max(
            epoch,
            image_epoch,
            noise_severity_max,
        )
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            if v2_noise_training:
                primary_image, primary_types, primary_severities = (
                    make_random_medical_noise_view(
                        image,
                        dataset_name,
                        noise_types=train_noise_types,
                        severity_min=noise_severity_min,
                        severity_max=current_severity_max,
                        apply_probability=primary_noise_probability,
                        noise_weights=train_noise_weights,
                    )
                )
                primary_image = preserve_lesion_contrast(
                    image,
                    primary_image,
                    mask,
                    min_retention=min_lesion_contrast_retention,
                )
                reference_images = []
                for _ in range(num_noise_views):
                    reference_image, reference_types, reference_severities = (
                        make_random_medical_noise_view(
                            primary_image,
                            dataset_name,
                            noise_types=train_noise_types,
                            severity_min=noise_severity_min,
                            severity_max=current_severity_max,
                            apply_probability=1.0,
                            noise_weights=train_noise_weights,
                        )
                    )
                    reference_image = preserve_lesion_contrast(
                        primary_image,
                        reference_image,
                        mask,
                        min_retention=min_lesion_contrast_retention,
                    )
                    reference_images.append(reference_image)
                    reference_noise_counts.update(reference_types)
                    sampled_severities.extend(reference_severities)
                primary_noise_counts.update(primary_types)
                sampled_severities.extend(primary_severities)
                model_input = primary_image
                reference_input = reference_images
            else:
                # V1 compatibility: a clean primary input and one
                # modality-specific auxiliary view.
                model_input = image
                reference_input = make_medical_noise_view(
                    image, dataset_name, severity=noise_severity
                )
            optimizer.zero_grad(set_to_none=True)
            patch_features, det_feature, auxiliary = model(
                model_input,
                reference_image=reference_input,
                text_embeddings=epoch_text_feature,
                return_aux=True,
            )
            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)
            for f in patch_features:
                # text-image alignment
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)  # backward
            if v2_noise_training:
                view_consistency_losses = torch.stack(
                    [
                        calculate_noise_consistency_loss(
                            auxiliary["primary_features"],
                            reference_features,
                            epoch_text_feature,
                        )
                        for reference_features in auxiliary[
                            "reference_feature_views"
                        ]
                    ]
                )
                consistency_loss = view_consistency_losses.mean()
                balance_loss = view_consistency_losses.var(unbiased=False)
            else:
                consistency_loss = calculate_noise_consistency_loss(
                    auxiliary["primary_features"],
                    auxiliary["reference_features"],
                    epoch_text_feature,
                )
                balance_loss = consistency_loss.new_zeros(())
            preservation_loss, boundary_loss = calculate_lesion_preservation_losses(
                patch_features,
                auxiliary["primary_features"],
                mask,
                boundary_margin=boundary_margin,
            )
            loss = loss + noise_consistency_weight * consistency_loss
            loss = loss + noise_balance_weight * balance_loss
            loss = loss + lesion_preservation_weight * preservation_loss
            loss = loss + boundary_contrast_weight * boundary_loss
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            consistency_loss_list.append(consistency_loss.item())
            balance_loss_list.append(balance_loss.item())
            preservation_loss_list.append(preservation_loss.item())
            boundary_loss_list.append(boundary_loss.item())
            scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        logger.info(
            "noise consistency: %.6f, noise balance: %.6f, "
            "lesion preservation: %.6f, boundary: %.6f",
            np.mean(consistency_loss_list),
            np.mean(balance_loss_list),
            np.mean(preservation_loss_list),
            np.mean(boundary_loss_list),
        )
        if v2_noise_training:
            logger.info(
                "V2 noise curriculum max: %.4f, sampled severity mean: %.4f, "
                "primary types: %s, reference types: %s",
                current_severity_max,
                np.mean(sampled_severities),
                dict(primary_noise_counts),
                dict(reference_noise_counts),
            )
        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
            "config": run_config,
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(
                model_dict,
                ckp_path,
            )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")
    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=5, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument("--text_lr", type=float, default=0.00001, help="learning rate for stage1")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument("--disable_patch_graph", action="store_true", help="disable patch-level graph refinement")
    parser.add_argument("--patch_graph_k", type=int, default=8)
    parser.add_argument("--patch_graph_alpha", type=float, default=0.7)
    parser.add_argument("--patch_graph_residual_weight", type=float, default=0.2)
    parser.add_argument("--disable_patch_graph_spatial", action="store_true", help="disable spatial edges in patch graph")
    parser.add_argument("--patch_graph_feature_temperature", type=float, default=0.2)
    parser.add_argument("--patch_graph_anomaly_temperature", type=float, default=0.2)
    parser.add_argument("--patch_graph_soft", action="store_true")
    parser.add_argument("--patch_graph_spectral_norm", action="store_true")
    parser.add_argument(
        "--graph_primary_only",
        action="store_true",
        help="use auxiliary views only to estimate uncertainty",
    )
    parser.add_argument("--noise_severity", type=float, default=0.06)
    parser.add_argument(
        "--train_noise_types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "enable V2 training with generic noise mechanisms; recommended: "
            + " ".join(GENERIC_MEDICAL_NOISE_TYPES)
        ),
    )
    parser.add_argument("--train_noise_weights", type=float, nargs="+", default=None)
    parser.add_argument("--primary_noise_probability", type=float, default=0.0)
    parser.add_argument("--noise_severity_min", type=float, default=0.0)
    parser.add_argument("--noise_severity_max", type=float, default=0.10)
    parser.add_argument("--num_noise_views", type=int, default=1)
    parser.add_argument("--noise_balance_weight", type=float, default=0.05)
    parser.add_argument(
        "--min_lesion_contrast_retention",
        type=float,
        default=0.7,
    )
    parser.add_argument("--noise_consistency_weight", type=float, default=0.1)
    parser.add_argument("--lesion_preservation_weight", type=float, default=0.1)
    parser.add_argument("--boundary_contrast_weight", type=float, default=0.05)
    parser.add_argument("--boundary_margin", type=float, default=0.2)

    args = parser.parse_args()
    if not 0.0 <= args.primary_noise_probability <= 1.0:
        parser.error("primary_noise_probability must be in [0, 1]")
    if args.noise_severity_min < 0 or args.noise_severity_max < args.noise_severity_min:
        parser.error("expected 0 <= noise_severity_min <= noise_severity_max")
    if args.num_noise_views < 1:
        parser.error("num_noise_views must be at least 1")
    if not 0.0 <= args.min_lesion_contrast_retention <= 1.0:
        parser.error("min_lesion_contrast_retention must be in [0, 1]")
    if args.train_noise_weights is not None:
        if args.train_noise_types is None:
            parser.error("train_noise_weights requires train_noise_types")
        if len(args.train_noise_weights) != len(args.train_noise_types):
            parser.error("train_noise_weights must match train_noise_types")
    if args.primary_noise_probability > 0 and args.train_noise_types is None:
        parser.error("primary noise training requires train_noise_types")
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # setup image feature extractor after surgery
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)
    # set up model for training
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        enable_patch_graph=not args.disable_patch_graph,
        patch_graph_k=args.patch_graph_k,
        patch_graph_alpha=args.patch_graph_alpha,
        patch_graph_residual_weight=args.patch_graph_residual_weight,
        patch_graph_use_spatial=not args.disable_patch_graph_spatial,
        patch_graph_feature_temperature=args.patch_graph_feature_temperature,
        patch_graph_anomaly_temperature=args.patch_graph_anomaly_temperature,
        patch_graph_soft=args.patch_graph_soft,
        patch_graph_spectral_norm=args.patch_graph_spectral_norm,
        graph_primary_only=args.graph_primary_only,
    ).to(device)
    model.eval()
    for parameter in model.clipmodel.parameters():
        parameter.requires_grad = False
    # set optimizer
    text_optimizer = torch.optim.Adam(
        model.text_trainable_parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        model.image_trainable_parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    # text_scheduler = MultiStepLR(text_optimizer, milestones=[400], gamma=0.1)
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)
    # ========================================================
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        try:
            text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        except ValueError:
            logger.info("skip text optimizer state because trainable parameters changed")
        text_start_epoch = checkpoint["epoch"]
        adapt_text = not (text_start_epoch == (args.text_epoch - 1))
    elif args.text_epoch == 0:
        adapt_text = False
    else:
        text_start_epoch = 0
        adapt_text = True  # check if text adapter is loaded
    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0], map_location=device)
        checkpoint_config = checkpoint.get("config", {})
        for config_name in (
            "patch_graph_soft",
            "patch_graph_spectral_norm",
            "graph_primary_only",
        ):
            if config_name in checkpoint_config and bool(
                checkpoint_config[config_name]
            ) != bool(getattr(args, config_name)):
                raise ValueError(
                    f"existing checkpoint architecture does not match "
                    f"--{config_name}; use a separate save_path"
                )
        image_start_epoch = checkpoint["epoch"]
        load_result = model.image_adapter.load_state_dict(
            checkpoint["image_adapter"], strict=False
        )
        if checkpoint_config and (
            load_result.missing_keys or load_result.unexpected_keys
        ):
            raise RuntimeError(
                "checkpoint/model architecture mismatch; missing keys: "
                f"{load_result.missing_keys}, unexpected keys: "
                f"{load_result.unexpected_keys}"
            )
        try:
            image_optimizer.load_state_dict(checkpoint["image_optimizer"])
        except ValueError:
            logger.info("skip image optimizer state because trainable parameters changed")
    else:
        image_start_epoch = 0
    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")
    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    # ========================================================
    # training
    if adapt_text:
        model = train_text_adapter(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            # scheduler=text_scheduler,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
        )
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()
    with torch.no_grad():
        if args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device, adapt_text=False
            )
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
    model = train_image_adapter(
        model=model,
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        dataset_name=args.dataset,
        noise_severity=args.noise_severity,
        noise_consistency_weight=args.noise_consistency_weight,
        noise_balance_weight=args.noise_balance_weight,
        lesion_preservation_weight=args.lesion_preservation_weight,
        boundary_contrast_weight=args.boundary_contrast_weight,
        boundary_margin=args.boundary_margin,
        train_noise_types=args.train_noise_types,
        train_noise_weights=args.train_noise_weights,
        primary_noise_probability=args.primary_noise_probability,
        noise_severity_min=args.noise_severity_min,
        noise_severity_max=args.noise_severity_max,
        num_noise_views=args.num_noise_views,
        min_lesion_contrast_retention=args.min_lesion_contrast_retention,
        run_config=vars(args).copy(),
        logger=logger,
    )


if __name__ == "__main__":
    main()
