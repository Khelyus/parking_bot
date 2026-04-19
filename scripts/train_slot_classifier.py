from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import random
import time


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training the slot classifier.") from exc
    return torch


def _require_torchvision():
    try:
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchvision is required for training the slot classifier.") from exc
    return models, transforms


def _require_pil_image():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for training the slot classifier.") from exc
    return Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PKLot slot classifier from COCO data.")
    parser.add_argument("--dataset-root", default=".", help="Project root that contains train/valid/test.")
    parser.add_argument("--train-json", default="train/_annotations.coco.json")
    parser.add_argument("--valid-json", default="valid/_annotations.coco.json")
    parser.add_argument("--test-json", default="test/_annotations.coco.json")
    parser.add_argument("--train-images", default="train")
    parser.add_argument("--valid-images", default="valid")
    parser.add_argument("--test-images", default="test")
    parser.add_argument("--output", default="models/pklot_slot_classifier.pt")
    parser.add_argument(
        "--base-model",
        default="",
        help="Optional checkpoint path to continue training from instead of ImageNet weights.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--train-samples-per-class", type=int, default=25000)
    parser.add_argument("--valid-samples-per-class", type=int, default=5000)
    parser.add_argument("--test-samples-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Number of PyTorch CPU threads. 0 means auto-detect from the machine.",
    )
    parser.add_argument(
        "--cpu-interop-threads",
        type=int,
        default=0,
        help="Number of PyTorch CPU interop threads. 0 picks a conservative auto value.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--progress-batches", type=int, default=0)
    parser.add_argument("--epoch-checkpoint-dir", default="")
    parser.add_argument(
        "--freeze-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the MobileNet feature extractor and train the classification head only.",
    )
    return parser.parse_args()


class CocoSlotCropDataset:
    CATEGORY_TO_LABEL = {
        1: 0,  # space-empty
        2: 1,  # space-occupied
    }

    def __init__(
        self,
        *,
        annotations_path: Path,
        images_dir: Path,
        image_size: int,
        samples_per_class: int | None,
        seed: int,
        training: bool,
    ) -> None:
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self.image_size = image_size
        self.training = training
        self._samples = self._load_samples(samples_per_class=samples_per_class, seed=seed)

        _models, transforms = _require_torchvision()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if training:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    @staticmethod
    def _normalize_sample_limit(value: int | None) -> int | None:
        if value is None:
            return None
        return None if value <= 0 else value

    def _load_samples(self, *, samples_per_class: int | None, seed: int) -> list[tuple[Path, tuple[float, float, float, float], int]]:
        samples_per_class = self._normalize_sample_limit(samples_per_class)
        payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        image_by_id = {
            int(item["id"]): self.images_dir / str(item["file_name"])
            for item in payload.get("images", [])
        }
        grouped: dict[int, list[tuple[Path, tuple[float, float, float, float], int]]] = defaultdict(list)

        for annotation in payload.get("annotations", []):
            category_id = int(annotation.get("category_id", -1))
            if category_id not in self.CATEGORY_TO_LABEL:
                continue
            image_path = image_by_id.get(int(annotation["image_id"]))
            if image_path is None or not image_path.exists():
                continue
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = [float(item) for item in bbox]
            if width <= 2 or height <= 2:
                continue
            grouped[self.CATEGORY_TO_LABEL[category_id]].append(
                (image_path, (x, y, width, height), self.CATEGORY_TO_LABEL[category_id])
            )

        rng = random.Random(seed)
        samples: list[tuple[Path, tuple[float, float, float, float], int]] = []
        for label in sorted(grouped):
            label_samples = grouped[label]
            rng.shuffle(label_samples)
            if samples_per_class is not None:
                label_samples = label_samples[:samples_per_class]
            samples.extend(label_samples)
        rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        Image = _require_pil_image()
        image_path, bbox, label = self._samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = self._crop_slot(image, bbox)
            tensor = self.transform(crop)
        return tensor, label

    def _crop_slot(self, image, bbox: tuple[float, float, float, float]):
        x, y, width, height = bbox
        padding_x = max(3.0, width * 0.18)
        padding_y_top = max(3.0, height * 0.18)
        padding_y_bottom = max(3.0, height * 0.12)
        x1 = max(0, int(round(x - padding_x)))
        y1 = max(0, int(round(y - padding_y_top)))
        x2 = min(image.width, int(round(x + width + padding_x)))
        y2 = min(image.height, int(round(y + height + padding_y_bottom)))
        return image.crop((x1, y1, x2, y2))


def _build_model(
    *,
    freeze_features: bool,
    base_model_path: Path | None = None,
):
    torch = _require_torch()
    models, _transforms = _require_torchvision()
    checkpoint = None
    weights = None
    if base_model_path is not None and base_model_path.exists():
        checkpoint = torch.load(base_model_path, map_location="cpu")
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, 2)
    if checkpoint is not None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    return model


def _accuracy(logits, targets) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).sum().item()) / max(1, int(targets.numel()))


def _run_epoch(
    *,
    model,
    loader,
    criterion,
    optimizer,
    device,
    training: bool,
    emit=None,
    stage: str = "train",
    progress_batches: int = 0,
) -> tuple[float, float]:
    torch = _require_torch()
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = max(1, len(loader))
    started = time.perf_counter()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size

        if emit is not None and progress_batches > 0 and batch_index % progress_batches == 0:
            elapsed = max(0.001, time.perf_counter() - started)
            batches_left = max(0, total_batches - batch_index)
            eta_seconds = elapsed / batch_index * batches_left
            running_loss = total_loss / max(1, total_items)
            running_accuracy = total_correct / max(1, total_items)
            emit(
                f"{stage}_progress batch={batch_index}/{total_batches} "
                f"loss={running_loss:.4f} acc={running_accuracy:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s"
            )

    if total_items == 0:
        return (0.0, 0.0)
    return (total_loss / total_items, total_correct / total_items)


def _build_logger(log_file: str | Path | None):
    log_path = Path(log_file).resolve() if log_file else None

    def emit(message: str) -> None:
        print(message, flush=True)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    return emit


def _save_checkpoint(
    checkpoint: dict[str, object],
    output_path: Path,
    *,
    epoch_checkpoint_dir: str | Path | None,
    epoch: int | None = None,
) -> None:
    torch = _require_torch()
    torch.save(checkpoint, output_path)
    if not epoch_checkpoint_dir or epoch is None:
        return
    checkpoint_dir = Path(epoch_checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = checkpoint_dir / f"{output_path.stem}.epoch_{epoch}.pt"
    torch.save(checkpoint, epoch_path)


def main() -> None:
    args = parse_args()
    torch = _require_torch()
    emit = _build_logger(args.log_file)

    dataset_root = Path(args.dataset_root).resolve()
    train_json = (dataset_root / args.train_json).resolve()
    valid_json = (dataset_root / args.valid_json).resolve()
    test_json = (dataset_root / args.test_json).resolve()
    train_images = (dataset_root / args.train_images).resolve()
    valid_images = (dataset_root / args.valid_images).resolve()
    test_images = (dataset_root / args.test_images).resolve()
    output_path = (dataset_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_model_path = (dataset_root / args.base_model).resolve() if args.base_model else None

    if args.device == "cpu":
        cpu_threads = max(1, args.cpu_threads or (os.cpu_count() or 1))
        cpu_interop_threads = max(1, args.cpu_interop_threads or min(4, cpu_threads))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(cpu_interop_threads)
        except RuntimeError:
            pass
        emit(
            f"cpu_threads={torch.get_num_threads()} "
            f"cpu_interop_threads={getattr(torch, 'get_num_interop_threads', lambda: 'n/a')()}"
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_dataset = CocoSlotCropDataset(
        annotations_path=train_json,
        images_dir=train_images,
        image_size=args.image_size,
        samples_per_class=args.train_samples_per_class,
        seed=args.seed,
        training=True,
    )
    valid_dataset = CocoSlotCropDataset(
        annotations_path=valid_json,
        images_dir=valid_images,
        image_size=args.image_size,
        samples_per_class=args.valid_samples_per_class,
        seed=args.seed + 1,
        training=False,
    )
    test_dataset = None
    if test_json.exists() and test_images.exists():
        test_dataset = CocoSlotCropDataset(
            annotations_path=test_json,
            images_dir=test_images,
            image_size=args.image_size,
            samples_per_class=args.test_samples_per_class,
            seed=args.seed + 2,
            training=False,
        )

    loader_kwargs = {
        "batch_size": args.batch,
        "num_workers": max(0, args.workers),
        "pin_memory": args.device != "cpu",
        "persistent_workers": max(0, args.workers) > 0,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = (
        torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    device = torch.device(args.device)
    model = _build_model(
        freeze_features=args.freeze_features,
        base_model_path=base_model_path,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    emit(
        f"Training slot classifier on {len(train_dataset)} train samples and "
        f"{len(valid_dataset)} validation samples."
    )
    emit(
        f"settings base_model={base_model_path or 'imagenet'} "
        f"batch={args.batch} workers={args.workers} epochs={args.epochs} "
        f"device={args.device} freeze_features={args.freeze_features}"
    )
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            emit=emit,
            stage="train",
            progress_batches=args.progress_batches,
        )
        valid_loss, valid_accuracy = _run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
            emit=emit,
            stage="valid",
            progress_batches=args.progress_batches,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
            }
        )
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())

        emit(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={valid_loss:.4f} val_acc={valid_accuracy:.4f} "
            f"time={time.perf_counter() - epoch_started:.1f}s"
        )

        epoch_checkpoint = {
            "arch": "mobilenet_v3_small",
            "class_names": ["space-empty", "space-occupied"],
            "image_size": args.image_size,
            "freeze_features": args.freeze_features,
            "model_state_dict": best_state_dict,
            "best_valid_accuracy": best_accuracy,
            "history": history,
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
            "test_samples": len(test_dataset) if test_dataset is not None else 0,
            "test_accuracy": None,
            "test_loss": None,
        }
        _save_checkpoint(
            epoch_checkpoint,
            output_path,
            epoch_checkpoint_dir=args.epoch_checkpoint_dir,
            epoch=epoch,
        )

    test_loss = None
    test_accuracy = None
    if test_loader is not None:
        model.load_state_dict(best_state_dict)
        test_loss, test_accuracy = _run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )
        emit(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")

    checkpoint = {
        "arch": "mobilenet_v3_small",
        "class_names": ["space-empty", "space-occupied"],
        "image_size": args.image_size,
        "freeze_features": args.freeze_features,
        "model_state_dict": best_state_dict,
        "best_valid_accuracy": best_accuracy,
        "history": history,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
    }
    _save_checkpoint(checkpoint, output_path, epoch_checkpoint_dir=None, epoch=None)
    emit(
        f"Saved slot classifier to {output_path} "
        f"(best_val_acc={best_accuracy:.4f}, total_time={time.perf_counter() - start_time:.1f}s)"
    )


if __name__ == "__main__":
    main()
