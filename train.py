import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from lodo_utils import (
    DEFAULT_LODO_DATASETS,
    DatasetWithName,
    HomogeneousDatasetBatchSampler,
    configure_lodo_datasets,
)
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    get_classification_text_embedding,
    calculate_image_logits,
    calculate_similarity_map,
    calculate_seg_loss,
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
    training_setup: dict = None,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]
            dataset_names = input_data.get(
                "dataset_name", [dataset_name] * len(class_names)
            )
            has_mask = input_data.get("has_mask")
            if has_mask is not None:
                valid_mask = has_mask.to(device=device, dtype=torch.bool)
                if not valid_mask.any():
                    continue
                valid_indices = valid_mask.nonzero(as_tuple=False).flatten().tolist()
                image = image[valid_mask]
                mask = mask[valid_mask]
                class_names = [class_names[index] for index in valid_indices]
                dataset_names = [dataset_names[index] for index in valid_indices]

            # forward text
            epoch_text_feature_dict = {}
            sample_keys = list(zip(dataset_names, class_names))
            for sample_dataset, class_name in set(sample_keys):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, sample_dataset, class_name, device
                )
                epoch_text_feature_dict[(sample_dataset, class_name)] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[key] for key in sample_keys],
                dim=0,
            )  # bs,768,2
            # forward image
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
                "training_setup": training_setup,
            },
            ckp_path,
        )
    return adapted_model


def train_image_adapter(
    model: nn.Module,
    localization_text_embeddings: dict,
    classification_text_embeddings: dict,
    train_loader: DataLoader,
    localization_optimizer: torch.optim.Optimizer,
    det_optimizer: torch.optim.Optimizer,
    localization_scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
    default_dataset_name: str,
    classification_loss_weight: float,
    localization_loss_weight: float,
    det_temperature: float,
    training_setup: dict = None,
):
    # Keep the frozen CLIP backbone in eval mode while enabling det-head dropout.
    model.image_adapter["det_head"].train()
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        classification_loss_list = []
        localization_loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape
            # forward text
            class_names = input_data["class_name"]
            dataset_names = input_data.get(
                "dataset_name", [default_dataset_name] * len(class_names)
            )
            has_mask = input_data.get("has_mask")
            if has_mask is None:
                has_mask = torch.ones(len(class_names), dtype=torch.bool, device=device)
            else:
                has_mask = has_mask.to(device=device, dtype=torch.bool)
            localization_text_feature = torch.stack(
                [
                    localization_text_embeddings[(sample_dataset, class_name)]
                    for sample_dataset, class_name in zip(dataset_names, class_names)
                ],
                dim=0,
            )
            classification_text_feature = torch.stack(
                [
                    classification_text_embeddings[(sample_dataset, class_name)]
                    for sample_dataset, class_name in zip(dataset_names, class_names)
                ],
                dim=0,
            )

            # forward image
            patch_features, det_feature = model(image)
            # The classification branch consumes only the independent CLS
            # feature; the localization branch consumes only patch features.
            image_logits = calculate_image_logits(
                det_feature,
                classification_text_feature,
                temperature=det_temperature,
            )
            classification_loss = F.cross_entropy(image_logits, label.long())
            loss = classification_loss_weight * classification_loss
            localization_loss = None
            if has_mask.any():
                # Graph-refined features keep their level axis:
                # (bs, num_levels, patch_num, 768). Similarity maps are
                # calculated independently and averaged by the utility.
                patch_preds = calculate_similarity_map(
                    patch_features[has_mask],
                    localization_text_feature[has_mask],
                    img_size,
                )
                localization_loss = calculate_seg_loss(
                    patch_preds, mask[has_mask]
                )
                loss += localization_loss_weight * localization_loss
            localization_optimizer.zero_grad()
            det_optimizer.zero_grad()
            loss.backward()
            localization_optimizer.step()
            det_optimizer.step()
            loss_list.append(loss.item())
            classification_loss_list.append(classification_loss.item())
            if localization_loss is not None:
                localization_loss_list.append(localization_loss.item())
            localization_scheduler.step()
        logger.info(
            "loss: %.6f, classification_loss: %.6f, localization_loss: %s, "
            "localization_lr: %.3e, det_lr: %.3e",
            np.mean(loss_list),
            np.mean(classification_loss_list),
            (
                f"{np.mean(localization_loss_list):.6f}"
                if localization_loss_list
                else "N/A"
            ),
            localization_optimizer.param_groups[0]["lr"],
            det_optimizer.param_groups[0]["lr"],
        )
        # save checkpoint
        det_head = model.image_adapter["det_head"]
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "localization_optimizer": localization_optimizer.state_dict(),
            "det_optimizer": det_optimizer.state_dict(),
            "localization_scheduler": localization_scheduler.state_dict(),
            "training_setup": training_setup,
            "classification_branch": "residual_bottleneck_cls_v2",
            "classification_text_branch": "frozen_clip_medical_v1",
            "localization_fusion": "multilevel_score_mean_v2",
            "patch_graph_config": model.patch_graph_config(),
            "det_head_config": {
                "hidden_dim": det_head.down.out_features,
                "dropout": det_head.dropout.p,
                "residual_scale": det_head.residual_scale,
            },
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(
                model_dict,
                ckp_path,
            )
    return model


def get_training_text_embeddings(model, dataset_names, device, adapt_text=True):
    text_embeddings = {}
    for dataset_name in dataset_names:
        dataset_embeddings = get_adapted_text_embedding(
            model, dataset_name, device, adapt_text=adapt_text
        )
        for class_name, embedding in dataset_embeddings.items():
            text_embeddings[(dataset_name, class_name)] = embedding
    return text_embeddings


def get_training_classification_text_embeddings(model, dataset_names, device):
    text_embeddings = {}
    for dataset_name in dataset_names:
        dataset_embeddings = get_classification_text_embedding(
            model,
            dataset_name,
            device,
        )
        for class_name, embedding in dataset_embeddings.items():
            text_embeddings[(dataset_name, class_name)] = embedding
    return text_embeddings


def validate_checkpoint_setup(checkpoint, training_setup):
    saved_setup = checkpoint.get("training_setup")
    requires_exact_setup = training_setup.get("mode") == "leave_one_out"
    if (requires_exact_setup or saved_setup is not None) and saved_setup != training_setup:
        raise ValueError(
            f"Checkpoint training setup {saved_setup} does not match "
            f"the requested setup {training_setup}"
        )


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
        "--leave_out",
        type=str,
        default=None,
        metavar="DATASET",
        help="hold out one dataset and train on the others in --datasets",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_LODO_DATASETS,
        help="dataset pool for leave-one-dataset-out training",
    )
    parser.add_argument(
        "--data_path",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="override a dataset root; repeat once per dataset when needed",
    )
    parser.add_argument(
        "--maskless_datasets",
        type=str,
        nargs="*",
        default=["Chest", "OCT2017", "HIS"],
        help="datasets without pixel masks; they use image-level loss only",
    )
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
    parser.add_argument(
        "--image_lr",
        type=float,
        default=0.0005,
        help="learning rate for the stage2 localization branch",
    )
    parser.add_argument(
        "--det_lr",
        type=float,
        default=0.0001,
        help="learning rate for the residual image-classification head",
    )
    parser.add_argument(
        "--det_weight_decay",
        type=float,
        default=0.0001,
        help="AdamW weight decay for the residual image-classification head",
    )
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
    parser.add_argument("--classification_loss_weight", type=float, default=1.0)
    parser.add_argument("--localization_loss_weight", type=float, default=1.0)
    parser.add_argument("--det_temperature", type=float, default=100.0)
    parser.add_argument("--det_hidden_dim", type=int, default=128)
    parser.add_argument("--det_dropout", type=float, default=0.1)
    parser.add_argument("--det_residual_scale", type=float, default=1.0)
    parser.add_argument("--disable_patch_graph", action="store_true", help="disable patch-level graph refinement")
    parser.add_argument("--patch_graph_k", type=int, default=8)
    parser.add_argument("--patch_graph_alpha", type=float, default=0.7)
    parser.add_argument("--patch_graph_residual_weight", type=float, default=0.7)
    parser.add_argument("--disable_patch_graph_spatial", action="store_true", help="disable spatial edges in patch graph")

    args = parser.parse_args()
    if args.classification_loss_weight < 0:
        parser.error("--classification_loss_weight must be non-negative")
    if args.localization_loss_weight < 0:
        parser.error("--localization_loss_weight must be non-negative")
    if args.det_temperature <= 0:
        parser.error("--det_temperature must be positive")
    if args.image_lr <= 0:
        parser.error("--image_lr must be positive")
    if args.det_lr <= 0:
        parser.error("--det_lr must be positive")
    if args.det_weight_decay < 0:
        parser.error("--det_weight_decay must be non-negative")
    if args.det_hidden_dim <= 0:
        parser.error("--det_hidden_dim must be positive")
    if not 0.0 <= args.det_dropout < 1.0:
        parser.error("--det_dropout must be in [0, 1)")
    if args.det_residual_scale < 0:
        parser.error("--det_residual_scale must be non-negative")
    if args.leave_out is not None:
        if len(args.datasets) != len(set(args.datasets)):
            parser.error("--datasets must not contain duplicates")
        if args.leave_out not in args.datasets:
            parser.error("--leave_out must be one of --datasets")
        training_dataset_names = [
            dataset_name
            for dataset_name in args.datasets
            if dataset_name != args.leave_out
        ]
        try:
            configure_lodo_datasets(training_dataset_names, args.data_path)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        training_setup = {
            "mode": "leave_one_out",
            "held_out_dataset": args.leave_out,
            "training_datasets": training_dataset_names,
            "maskless_datasets": args.maskless_datasets,
        }
    else:
        training_dataset_names = [args.dataset]
        training_setup = {
            "mode": "single_dataset",
            "held_out_dataset": None,
            "training_datasets": training_dataset_names,
            "maskless_datasets": args.maskless_datasets,
        }
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
        det_hidden_dim=args.det_hidden_dim,
        det_dropout=args.det_dropout,
        det_residual_scale=args.det_residual_scale,
    ).to(device)
    model.eval()
    # set optimizer
    text_optimizer = torch.optim.Adam(
        model.text_trainable_parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    localization_optimizer = torch.optim.Adam(
        model.localization_trainable_parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    det_optimizer = torch.optim.AdamW(
        model.classification_trainable_parameters(),
        lr=args.det_lr,
        betas=(0.9, 0.999),
        weight_decay=args.det_weight_decay,
    )
    # text_scheduler = MultiStepLR(text_optimizer, milestones=[400], gamma=0.1)
    localization_scheduler = MultiStepLR(
        localization_optimizer, milestones=[16000, 32000], gamma=0.5
    )
    # ========================================================
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        validate_checkpoint_setup(checkpoint, training_setup)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"], strict=False)
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
        checkpoint = torch.load(file[0])
        validate_checkpoint_setup(checkpoint, training_setup)
        if checkpoint.get("classification_branch") != "residual_bottleneck_cls_v2":
            raise ValueError(
                "The existing image checkpoint uses a different classification "
                "head. Use a new --save_path for the residual det-head experiment."
            )
        if checkpoint.get("classification_text_branch") != "frozen_clip_medical_v1":
            raise ValueError(
                "The existing image checkpoint was trained with a different "
                "classification text branch. Use a new --save_path."
            )
        if checkpoint.get("localization_fusion") != "multilevel_score_mean_v2":
            raise ValueError(
                "The existing image checkpoint was trained with feature-level "
                "localization fusion. Use a new --save_path for the multilevel "
                "score-fusion experiment."
            )
        requested_patch_graph_config = model.patch_graph_config()
        if checkpoint.get("patch_graph_config") != requested_patch_graph_config:
            raise ValueError(
                "The existing image checkpoint Patch Graph configuration "
                f"{checkpoint.get('patch_graph_config')} does not match "
                f"{requested_patch_graph_config}. Use the training-time Graph "
                "arguments."
            )
        requested_det_head_config = {
            "hidden_dim": args.det_hidden_dim,
            "dropout": args.det_dropout,
            "residual_scale": args.det_residual_scale,
        }
        if checkpoint.get("det_head_config") != requested_det_head_config:
            raise ValueError(
                "The existing image checkpoint det-head configuration "
                f"{checkpoint.get('det_head_config')} does not match "
                f"{requested_det_head_config}. Use matching arguments or a new "
                "--save_path."
            )
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"], strict=False)
        try:
            localization_optimizer.load_state_dict(
                checkpoint["localization_optimizer"]
            )
            det_optimizer.load_state_dict(checkpoint["det_optimizer"])
            localization_scheduler.load_state_dict(
                checkpoint["localization_scheduler"]
            )
        except (KeyError, ValueError):
            logger.info(
                "skip image optimizer state because trainable parameters changed"
            )
    else:
        image_start_epoch = 0
    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")
    if args.leave_out is None:
        text_dataset, image_dataset = get_dataset(
            args.dataset,
            args.img_size,
            args.training_mode,
            args.shot,
            "train",
            logger,
        )
        if args.dataset in args.maskless_datasets:
            text_dataset = DatasetWithName(
                text_dataset, args.dataset, has_pixel_masks=False
            )
            image_dataset = DatasetWithName(
                image_dataset, args.dataset, has_pixel_masks=False
            )
    else:
        text_datasets = []
        image_datasets = []
        for dataset_name in training_dataset_names:
            source_text_dataset, source_image_dataset = get_dataset(
                dataset_name,
                args.img_size,
                args.training_mode,
                args.shot,
                "train",
                logger,
            )
            has_pixel_masks = dataset_name not in args.maskless_datasets
            text_datasets.append(
                DatasetWithName(
                    source_text_dataset,
                    dataset_name,
                    has_pixel_masks=has_pixel_masks,
                )
            )
            image_datasets.append(
                DatasetWithName(
                    source_image_dataset,
                    dataset_name,
                    has_pixel_masks=has_pixel_masks,
                )
            )
            logger.info(
                "leave-one-out source %s: %d samples",
                dataset_name,
                len(source_image_dataset),
            )
        text_dataset = ConcatDataset(text_datasets)
        image_dataset = ConcatDataset(image_datasets)
        logger.info("leave-one-out held dataset: %s", args.leave_out)
    if args.leave_out is None:
        text_dataloader = torch.utils.data.DataLoader(
            text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
        )
    else:
        text_batch_sampler = HomogeneousDatasetBatchSampler(
            text_dataset,
            batch_size=args.text_batch_size,
            shuffle=True,
        )
        text_dataloader = torch.utils.data.DataLoader(
            text_dataset,
            batch_sampler=text_batch_sampler,
            **kwargs,
        )
    logger.info("loading image adaptation dataset ...")
    if args.leave_out is None:
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
        )
    else:
        image_batch_sampler = HomogeneousDatasetBatchSampler(
            image_dataset,
            batch_size=args.image_batch_size,
            shuffle=True,
        )
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset,
            batch_sampler=image_batch_sampler,
            **kwargs,
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
            training_setup=training_setup,
        )
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()
    with torch.no_grad():
        localization_text_embeddings = get_training_text_embeddings(
            model,
            training_dataset_names,
            device,
            adapt_text=args.text_epoch != 0,
        )
        classification_text_embeddings = (
            get_training_classification_text_embeddings(
                model,
                training_dataset_names,
                device,
            )
        )
    model = train_image_adapter(
        model=model,
        localization_text_embeddings=localization_text_embeddings,
        classification_text_embeddings=classification_text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        localization_optimizer=localization_optimizer,
        det_optimizer=det_optimizer,
        localization_scheduler=localization_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
        default_dataset_name=args.dataset,
        classification_loss_weight=args.classification_loss_weight,
        localization_loss_weight=args.localization_loss_weight,
        det_temperature=args.det_temperature,
        training_setup=training_setup,
    )


if __name__ == "__main__":
    main()
