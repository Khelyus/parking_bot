from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for slot-classifier fine-tuning.") from exc
    return torch


def _require_torchvision():
    try:
        from torchvision import datasets, models, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchvision is required for slot-classifier fine-tuning.") from exc
    return datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the slot classifier on Ufanet calibration crops.")
    parser.add_argument("--dataset-dir", default="runtime/slot_calibration_dataset")
    parser.add_argument("--base-model", default="models/pklot_slot_classifier.pt")
    parser.add_argument("--output", default="models/pklot_slot_classifier.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--weight-decay", type=float, default=0.00005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    return parser.parse_args()


def _build_model(base_model_path: Path):
    torch = _require_torch()
    _datasets, models, _transforms = _require_torchvision()
    checkpoint = torch.load(base_model_path, map_location="cpu")
    class_names = checkpoint.get("class_names", ["space-empty", "space-occupied"])
    image_size = int(checkpoint.get("image_size", 224))
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, class_names, image_size, checkpoint


def _run_epoch(*, model, loader, criterion, optimizer, device, training: bool) -> tuple[float, float]:
    torch = _require_torch()
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for images, labels in loader:
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

    if total_items == 0:
        return (0.0, 0.0)
    return (total_loss / total_items, total_correct / total_items)


def main() -> None:
    args = parse_args()
    torch = _require_torch()
    datasets, _models, transforms = _require_torchvision()

    dataset_dir = Path(args.dataset_dir).resolve()
    base_model_path = Path(args.base_model).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, class_names, checkpoint_image_size, base_checkpoint = _build_model(base_model_path)
    image_size = args.image_size or checkpoint_image_size
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    if len(dataset) < 10:
        raise RuntimeError(f"Calibration dataset is too small: {len(dataset)} samples found in {dataset_dir}")
    if len(dataset.classes) != 2:
        raise RuntimeError(f"Expected 2 classes in {dataset_dir}, found {dataset.classes}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = max(2, int(len(indices) * args.val_split))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    if not train_indices or not val_indices:
        raise RuntimeError("Calibration split produced an empty train or validation set.")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    loader_kwargs = {
        "batch_size": args.batch,
        "num_workers": max(0, args.workers),
        "pin_memory": args.device != "cpu",
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    print(
        f"Fine-tuning on {len(train_dataset)} train and {len(val_dataset)} val calibration crops "
        f"from {dataset_dir}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_loss": val_loss,
                "valid_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    output_checkpoint = dict(base_checkpoint)
    output_checkpoint.update(
        {
            "class_names": class_names,
            "image_size": image_size,
            "model_state_dict": best_state_dict,
            "ufanet_finetune": {
                "dataset_dir": str(dataset_dir),
                "train_samples": len(train_dataset),
                "valid_samples": len(val_dataset),
                "best_valid_accuracy": best_accuracy,
                "history": history,
            },
        }
    )
    torch.save(output_checkpoint, output_path)
    print(
        f"Saved fine-tuned classifier to {output_path} "
        f"(best_val_acc={best_accuracy:.4f}, elapsed={time.perf_counter() - started:.1f}s)"
    )


if __name__ == "__main__":
    main()
