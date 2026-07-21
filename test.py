import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils import (
    make_medical_noise_view,
    noise_model_for_dataset,
    setup_seed,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
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


def checkpoint_sort_key(path):
    """Sort numbered image checkpoints by epoch instead of alphabetically."""
    stem = os.path.splitext(os.path.basename(path))[0]
    epoch_suffix = stem.rsplit("_", 1)[-1]
    if epoch_suffix.isdigit():
        return (0, int(epoch_suffix))
    return (1, stem)


def severity_tag(severity):
    return f"{severity:g}".replace(".", "p")


def noise_type_or_none(noise_type):
    return None if noise_type.lower() == "auto" else noise_type


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
    noise_severity: float = 0.06,
    test_noise_severity: float = 0.0,
    test_noise_type: str = "auto",
    probe_noise_type: str = "auto",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        # Robustness evaluation: corrupt the primary test input itself. This
        # is separate from the auxiliary noise view used by the model below.
        if test_noise_severity > 0:
            image = make_medical_noise_view(
                image,
                dataset,
                severity=test_noise_severity,
                noise_type=noise_type_or_none(test_noise_type),
            )
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_feature = class_text_embeddings
        # Estimate patch noise sensitivity from a second, mask-aligned view of
        # the (possibly corrupted) primary input.
        reference_image = make_medical_noise_view(
            image,
            dataset,
            severity=noise_severity,
            noise_type=noise_type_or_none(probe_noise_type),
        )
        patch_features, det_feature = model(
            image,
            reference_image=reference_image,
            text_embeddings=epoch_text_feature,
        )
        # calculate similarity and get prediction
        # cls_preds = []
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            # f: bs,patch_num,768
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Testing")
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
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument(
        "--image_checkpoint",
        type=str,
        default="image_adapter.pth",
        help="image checkpoint filename or glob pattern under save_path",
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
    parser.add_argument(
        "--noise_severity",
        type=float,
        default=0.06,
        help="noise magnitude for the model's auxiliary uncertainty view",
    )
    parser.add_argument(
        "--test_noise_severity",
        type=float,
        default=0.0,
        help="modality-aware corruption magnitude applied to the primary test input",
    )
    parser.add_argument(
        "--test_noise_type",
        type=str,
        default="auto",
        help="primary corruption type; auto selects the target modality",
    )
    parser.add_argument(
        "--probe_noise_type",
        type=str,
        default="auto",
        help="auxiliary probe type; auto selects the target modality",
    )

    args = parser.parse_args()
    if args.noise_severity < 0 or args.test_noise_severity < 0:
        parser.error("noise severities must be non-negative")
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
        patch_graph_feature_temperature=args.patch_graph_feature_temperature,
        patch_graph_anomaly_temperature=args.patch_graph_anomaly_temperature,
        patch_graph_soft=args.patch_graph_soft,
        patch_graph_spectral_norm=args.patch_graph_spectral_norm,
        graph_primary_only=args.graph_primary_only,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    checkpoint_pattern = args.image_checkpoint
    if not os.path.isabs(checkpoint_pattern):
        checkpoint_pattern = os.path.join(args.save_path, checkpoint_pattern)
    files = sorted(glob(checkpoint_pattern), key=checkpoint_sort_key)
    assert len(files) > 0, (
        f"image adapter checkpoint not found: {checkpoint_pattern}"
    )
    logger.info("testing %d image checkpoint(s): %s", len(files), files)
    summary_rows = []
    for file in files:
        checkpoint = torch.load(file, map_location=device)
        checkpoint_config = checkpoint.get("config", {})
        for config_name in (
            "patch_graph_soft",
            "patch_graph_spectral_norm",
            "graph_primary_only",
        ):
            if config_name in checkpoint_config:
                expected = bool(checkpoint_config[config_name])
                actual = bool(getattr(args, config_name))
                if expected != actual:
                    raise ValueError(
                        f"checkpoint requires --{config_name}={expected}, "
                        f"but the test configuration uses {actual}"
                    )
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
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
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
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device, adapt_text=False
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
        class_results = []
        # Use exactly the same stochastic corruptions for every checkpoint so
        # epoch comparisons are not confounded by different random noise.
        setup_seed(args.seed)
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                    noise_severity=args.noise_severity,
                    test_noise_severity=args.test_noise_severity,
                    test_noise_type=args.test_noise_type,
                    probe_noise_type=args.probe_noise_type,
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
            )
            class_results.append(class_result_dict)
            df.loc[len(df)] = Series(class_result_dict)
        metric_columns = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
        # Average the original numeric dictionaries directly. This avoids
        # pandas-version-dependent object dtype inference in an initially
        # empty DataFrame.
        average_result = {
            metric_name: float(
                np.mean([float(result[metric_name]) for result in class_results])
            )
            for metric_name in metric_columns
        }
        average_row = {"class name": "Average", **average_result}
        df.loc[len(df)] = average_row
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
        noise_tag = severity_tag(args.test_noise_severity)
        test_noise_name = (
            noise_model_for_dataset(args.dataset)
            if args.test_noise_type == "auto"
            else args.test_noise_type
        )
        result_path = os.path.join(
            args.save_path,
            f"test_{args.dataset}_epoch_{test_epoch}_"
            f"{test_noise_name}_noise_{noise_tag}.csv",
        )
        df.to_csv(result_path, index=False)
        summary_row = {
            "checkpoint": os.path.basename(file),
            "epoch": int(test_epoch),
            "test noise type": test_noise_name,
            "test noise severity": args.test_noise_severity,
        }
        for metric_name in metric_columns:
            summary_row[metric_name] = average_result[metric_name]
        summary_rows.append(summary_row)
        logger.info("saved checkpoint results to %s", result_path)
        print(f"epoch {test_epoch} results saved to {result_path}")

    summary = DataFrame(summary_rows)
    summary_path = os.path.join(
        args.save_path,
        f"test_{args.dataset}_all_checkpoints_noise_"
        f"{test_noise_name}_{severity_tag(args.test_noise_severity)}.csv",
    )
    summary.to_csv(summary_path, index=False)
    logger.info("all checkpoint results:\n%s", summary.to_string(index=False))
    print(f"all checkpoint results saved to {summary_path}")


if __name__ == "__main__":
    main()
