import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from lodo_utils import DatasetWithName, configure_lodo_datasets
from forward_utils import (
    get_adapted_text_embedding,
    get_classification_text_embedding,
    calculate_image_probability,
    calculate_patch_image_probability,
    calculate_similarity_map,
    metrics_eval,
    visualize,
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


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens, _ = model(image)
        patch_tokens = [patch_tokens.reshape(-1, 768)]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
    model: nn.Module,
    localization_text_embeddings: torch.Tensor,
    classification_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
    image_score_source: str = "det",
    det_temperature: float = 100.0,
    image_pooling: str = "topk",
    image_topk_ratio: float = 0.01,
    image_quantile: float = 0.99,
    image_temperature: float = 1.0,
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    mask_validity = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        has_mask = input_data.get("has_mask")
        if has_mask is None:
            has_mask = torch.ones(len(label), dtype=torch.bool)
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        mask_validity.append(has_mask.cpu().numpy().astype(bool))
        labels.append(label)
        file_names.extend(file_name)
        # get text
        # forward image
        patch_features, det_feature = model(image)
        # calculate similarity and get prediction
        if image_score_source == "det":
            pred = calculate_image_probability(
                det_feature,
                classification_text_embeddings,
                temperature=det_temperature,
            )
        else:
            pred = calculate_patch_image_probability(
                patch_features,
                localization_text_embeddings,
                temperature=image_temperature,
                aggregation=image_pooling,
                topk_ratio=image_topk_ratio,
                quantile=image_quantile,
            )
        preds_image.append(pred.cpu().numpy())
        # patch_features: (bs, patch_num, 768), already fused across the 4 levels
        patch_pred = calculate_similarity_map(
            patch_features,
            localization_text_embeddings,
            img_size,
            test=True,
            domain=DOMAINS[dataset],
        )
        preds.append(patch_pred.cpu().numpy())
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    mask_validity = np.concatenate(mask_validity, axis=0)
    return masks, labels, preds, preds_image, mask_validity, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument(
        "--data_path",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="override the test dataset root",
    )
    parser.add_argument(
        "--maskless_datasets",
        type=str,
        nargs="*",
        default=["Chest", "OCT2017", "HIS"],
        help="datasets without pixel-level ground truth",
    )
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--image_score_source",
        type=str,
        default="det",
        choices=["det", "patch"],
        help="use the independent CLS branch or legacy patch aggregation",
    )
    parser.add_argument(
        "--det_temperature",
        type=float,
        default=100.0,
        help="positive scale applied to CLS/text cosine logits",
    )
    parser.add_argument("--det_hidden_dim", type=int, default=128)
    parser.add_argument("--det_dropout", type=float, default=0.1)
    parser.add_argument("--det_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--image_pooling",
        type=str,
        default="topk",
        choices=["mean", "topk", "max", "quantile"],
        help="aggregate patch anomaly probabilities into an image score",
    )
    parser.add_argument(
        "--image_topk_ratio",
        type=float,
        default=0.01,
        help="fraction of highest-scoring patches used by topk pooling",
    )
    parser.add_argument(
        "--image_quantile",
        type=float,
        default=0.99,
        help="quantile used by quantile pooling",
    )
    parser.add_argument(
        "--image_temperature",
        type=float,
        default=1.0,
        help="positive scale applied to patch/text cosine logits",
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="image_adapter.pth",
        help="image-adapter checkpoint name or absolute path",
    )
    parser.add_argument("--visualize", action="store_true")
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
    if args.det_temperature <= 0:
        parser.error("--det_temperature must be positive")
    if args.det_hidden_dim <= 0:
        parser.error("--det_hidden_dim must be positive")
    if not 0.0 <= args.det_dropout < 1.0:
        parser.error("--det_dropout must be in [0, 1)")
    if args.det_residual_scale < 0:
        parser.error("--det_residual_scale must be non-negative")
    try:
        configure_lodo_datasets([args.dataset], args.data_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
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
    # load checkpoints if exists
    checkpoint_file = args.checkpoint
    if not os.path.isabs(checkpoint_file):
        checkpoint_file = os.path.join(args.save_path, checkpoint_file)
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"image adapter checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    has_det_branch = any(
        key.startswith("det_head.") for key in checkpoint["image_adapter"]
    )
    if args.image_score_source == "det" and not has_det_branch:
        raise ValueError(
            "This checkpoint has no trained residual classification head. "
            "Retrain with a new --save_path or use --image_score_source patch."
        )
    if (
        args.image_score_source == "det"
        and checkpoint.get("classification_branch")
        != "residual_bottleneck_cls_v2"
    ):
        raise ValueError(
            "This checkpoint uses a different classification head. Retrain "
            "with the residual det head or use --image_score_source patch."
        )
    if (
        args.image_score_source == "det"
        and checkpoint.get("classification_text_branch")
        != "frozen_clip_medical_v1"
    ):
        raise ValueError(
            "This checkpoint was trained with a different classification text "
            "branch. Retrain with a new --save_path or use "
            "--image_score_source patch."
        )
    requested_det_head_config = {
        "hidden_dim": args.det_hidden_dim,
        "dropout": args.det_dropout,
        "residual_scale": args.det_residual_scale,
    }
    if (
        args.image_score_source == "det"
        and checkpoint.get("det_head_config") != requested_det_head_config
    ):
        raise ValueError(
            "Checkpoint det-head configuration "
            f"{checkpoint.get('det_head_config')} does not match "
            f"{requested_det_head_config}. Use the training-time det arguments."
        )
    training_setup = checkpoint.get("training_setup")
    if training_setup and training_setup.get("mode") == "leave_one_out":
        held_out_dataset = training_setup.get("held_out_dataset")
        if held_out_dataset != args.dataset:
            raise ValueError(
                f"Checkpoint held out {held_out_dataset!r}, but test dataset is "
                f"{args.dataset!r}"
            )
    model.image_adapter.load_state_dict(checkpoint["image_adapter"], strict=False)
    test_epoch = checkpoint["epoch"]

    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        text_checkpoint = torch.load(text_file[0], map_location=device)
        text_setup = text_checkpoint.get("training_setup")
        if (
            training_setup
            and training_setup.get("mode") == "leave_one_out"
            and text_setup != training_setup
        ):
            raise ValueError("Text and image checkpoints use different training folds")
        model.text_adapter.load_state_dict(
            text_checkpoint["text_adapter"], strict=False
        )
        adapt_text = True
    else:
        adapt_text = False

    logger.info("-----------------------------------------------")
    logger.info("load model from epoch %d", test_epoch)
    logger.info("training setup: %s", training_setup)
    logger.info("-----------------------------------------------")
    # ========================================================
    # load dataset
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "test",
        logger=logger,
    )
    if args.dataset in args.maskless_datasets:
        image_datasets = {
            class_name: DatasetWithName(
                image_dataset,
                args.dataset,
                has_pixel_masks=False,
            )
            for class_name, image_dataset in image_datasets.items()
        }
    with torch.no_grad():
        if adapt_text:
            localization_text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device
            )
        else:
            localization_text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device, adapt_text=False
            )
        classification_text_embeddings = get_classification_text_embedding(
            model,
            args.dataset,
            device,
        )
    # ========================================================
    df = DataFrame(
        columns=[
            "class name",
            "pixel AUC",
            "pixel AP",
            "image AUC",
            "image AP",
        ]
    )
    for class_name, image_dataset in image_datasets.items():
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        # ========================================================
        # testing
        with torch.no_grad():
            class_localization_text_embeddings = localization_text_embeddings[
                class_name
            ]
            class_classification_text_embeddings = classification_text_embeddings[
                class_name
            ]
            masks, labels, preds, preds_image, mask_validity, file_names = get_predictions(
                model=model,
                localization_text_embeddings=class_localization_text_embeddings,
                classification_text_embeddings=class_classification_text_embeddings,
                test_loader=image_dataloader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
                image_score_source=args.image_score_source,
                det_temperature=args.det_temperature,
                image_pooling=args.image_pooling,
                image_topk_ratio=args.image_topk_ratio,
                image_quantile=args.image_quantile,
                image_temperature=args.image_temperature,
            )
        # ========================================================
        if args.visualize:
            visualize(
                masks,
                preds,
                file_names,
                args.save_path,
                args.dataset,
                class_name=class_name,
            )
        class_result_dict = metrics_eval(
            masks,
            labels,
            preds,
            preds_image,
            class_name,
            domain=DOMAINS[args.dataset],
            pixel_validity=mask_validity,
        )
        df.loc[len(df)] = Series(class_result_dict)
    df.loc[len(df)] = df.drop(columns="class name").mean()
    df.loc[len(df) - 1, "class name"] = "Average"
    numeric_cols = [col for col in df.columns if col != "class name"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    logger.info(
        "final results:\n%s",
        df.to_string(
            index=False,
            justify="center",
            formatters={col: "{:.4f}".format for col in numeric_cols},
        ),
    )


if __name__ == "__main__":
    main()
