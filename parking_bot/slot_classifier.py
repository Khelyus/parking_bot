from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install torch before using the slot classifier."
        ) from exc
    return torch


def _require_torchvision_models() -> Any:
    try:
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is not installed. Install torchvision before using the slot classifier."
        ) from exc
    return models


def _require_torchvision_transforms() -> Any:
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is not installed. Install torchvision before using the slot classifier."
        ) from exc
    return transforms


def _require_pil_image() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is not installed. Install pillow before using the slot classifier."
        ) from exc
    return Image


@dataclass(slots=True)
class SlotPrediction:
    label: str
    confidence: float
    empty_probability: float
    occupied_probability: float


class SlotStatusClassifier:
    DEFAULT_CLASS_NAMES = ("space-empty", "space-occupied")

    def __init__(self, model_path: str | Path, image_size: int = 224) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._model: Any | None = None
        self._transform: Any | None = None
        self._class_names = list(self.DEFAULT_CLASS_NAMES)

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise RuntimeError(f"Slot classifier model was not found: {self.model_path}")

        torch = _require_torch()
        models = _require_torchvision_models()
        transforms = _require_torchvision_transforms()
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.image_size = int(checkpoint.get("image_size", self.image_size))
        class_names = checkpoint.get("class_names")
        if isinstance(class_names, (list, tuple)) and len(class_names) == 2:
            self._class_names = [str(item) for item in class_names]

        model = models.mobilenet_v3_small(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(input_features, 2)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._model = model
        return self._model

    def predict_image(self, image: Any) -> SlotPrediction:
        if image is None or getattr(image, "size", 0) == 0:
            raise RuntimeError("Slot classifier image is empty")

        torch = _require_torch()
        Image = _require_pil_image()
        model = self._ensure_model()
        if self._transform is None:
            raise RuntimeError("Slot classifier transform was not initialized")

        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)
        tensor = self._transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].tolist()

        empty_probability = float(probabilities[0])
        occupied_probability = float(probabilities[1])
        if occupied_probability > empty_probability:
            return SlotPrediction(
                label=self._class_names[1],
                confidence=occupied_probability,
                empty_probability=empty_probability,
                occupied_probability=occupied_probability,
            )
        return SlotPrediction(
            label=self._class_names[0],
            confidence=empty_probability,
            empty_probability=empty_probability,
            occupied_probability=occupied_probability,
        )
