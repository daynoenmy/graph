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
    configure_lodo_datasets,
)
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
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
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
    default_dataset_name: str,
    training_setup: dict = None,
):
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
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
            epoch_text_feature = torch.stack(
                [
                    text_embeddings[(sample_dataset, class_name)]
                    for sample_dataset, class_name in zip(dataset_names, class_names)
                ],
                dim=0,
            )

            # forward image
            patch_features, det_feature = model(image)
            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)
            if has_mask.any():
                # patch_features is already fused across the 4 levels:
                # (bs, patch_num, 768)
                patch_preds = calculate_similarity_map(
                    patch_features[has_mask], epoch_text_feature[has_mask], img_size
                )
                loss += calculate_seg_loss(patch_preds, mask[has_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
            "training_setup": training_setup,
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
    parser.add_argument("--patch_graph_residual_weight", type=float, default=0.7)
    parser.add_argument("--disable_patch_graph_spatial", action="store_true", help="disable spatial edges in patch graph")

    args = parser.parse_args()
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
    ).to(device)
    model.eval()
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
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"], strict=False)
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
            training_setup=training_setup,
        )
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()
    with torch.no_grad():
        text_embeddings = get_training_text_embeddings(
            model,
            training_dataset_names,
            device,
            adapt_text=args.text_epoch != 0,
        )
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
        logger=logger,
        default_dataset_name=args.dataset,
        training_setup=training_setup,
    )


if __name__ == "__main__":
    main()
