from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "Slot Dataset Reviewer"
CLASS_EMPTY = "space-empty"
CLASS_OCCUPIED = "space-occupied"
SIDE_PANEL_WIDTH = 280
PANEL_PADDING = 16
PANEL_BG = (96, 96, 96)
CANVAS_BG = (44, 44, 44)


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for the dataset reviewer.") from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review slot calibration crops and move/delete them with keyboard shortcuts."
    )
    parser.add_argument(
        "--dataset-dir",
        default="runtime/slot_calibration_dataset",
        help="Dataset root containing class folders like space-empty and space-occupied.",
    )
    parser.add_argument(
        "--start-folder",
        help="Optional initial working folder name or path. If omitted, the tool will ask in the console.",
    )
    parser.add_argument("--max-width", type=int, default=1400, help="Maximum preview width.")
    parser.add_argument("--max-height", type=int, default=1000, help="Maximum preview height.")
    return parser.parse_args()


@dataclass
class ReviewState:
    dataset_dir: Path
    working_dir: Path
    image_paths: list[Path]
    index: int = 0


def _resolve_dataset_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _list_reviewable_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory was not found: {dataset_dir}")
    dirs = [path for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
    reviewable = [path for path in dirs if any(file.suffix.lower() in IMAGE_EXTENSIONS for file in path.iterdir() if file.is_file())]
    if not reviewable:
        raise RuntimeError(f"No reviewable class folders were found in {dataset_dir}")
    return reviewable


def _choose_working_dir(dataset_dir: Path, raw_choice: str | None) -> Path:
    reviewable = _list_reviewable_dirs(dataset_dir)
    if raw_choice:
        candidate = Path(raw_choice)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            direct = (dataset_dir / raw_choice).resolve()
            resolved = direct if direct.exists() else (PROJECT_ROOT / raw_choice).resolve()
        if resolved in reviewable:
            return resolved
        raise RuntimeError(f"Working folder was not found among reviewable folders: {resolved}")

    print("Select working folder:")
    for index, path in enumerate(reviewable, start=1):
        count = len([file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS])
        print(f"  {index}. {path.name} | images={count}")

    while True:
        raw = input("Enter folder number or name: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            selected_index = int(raw) - 1
            if 0 <= selected_index < len(reviewable):
                return reviewable[selected_index]
        for path in reviewable:
            if raw == path.name:
                return path
        print("Invalid selection. Try again.")


def _collect_images(working_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in working_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    available_width = max(120, max_width - (SIDE_PANEL_WIDTH * 2))
    available_height = max(120, max_height)
    scale = min(available_width / width, available_height / height, 1.0)
    return max(scale, 0.1)


def _load_image(path: Path) -> Any:
    cv2 = _require_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image


def _put_panel_lines(
    canvas: Any,
    *,
    lines: list[str],
    start_x: int,
    start_y: int,
    color: tuple[int, int, int] = (245, 245, 245),
    line_height: int = 28,
    font_scale: float = 0.58,
) -> None:
    cv2 = _require_cv2()
    y = start_y
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += line_height


def _render(image: Any, *, scale: float, state: ReviewState) -> Any:
    cv2 = _require_cv2()
    scaled_width = max(1, round(image.shape[1] * scale))
    scaled_height = max(1, round(image.shape[0] * scale))
    resized = (
        cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        if scale != 1.0
        else image
    )

    canvas_height = max(scaled_height + (PANEL_PADDING * 2), 720)
    canvas_width = (SIDE_PANEL_WIDTH * 2) + scaled_width
    canvas = cv2.copyMakeBorder(
        resized,
        top=(canvas_height - scaled_height) // 2,
        bottom=canvas_height - scaled_height - ((canvas_height - scaled_height) // 2),
        left=SIDE_PANEL_WIDTH,
        right=SIDE_PANEL_WIDTH,
        borderType=cv2.BORDER_CONSTANT,
        value=CANVAS_BG,
    )
    canvas[:, :SIDE_PANEL_WIDTH] = PANEL_BG
    canvas[:, canvas_width - SIDE_PANEL_WIDTH : canvas_width] = PANEL_BG

    left_lines = [
        f"folder={state.working_dir.name}",
        f"image={state.index + 1}/{len(state.image_paths)}",
        f"name={state.image_paths[state.index].name}",
    ]
    right_lines = [
        "Controls",
        "Left / Right",
        "prev / next image",
        "",
        "Delete",
        "delete current file",
        "",
        "O",
        f"move to {CLASS_OCCUPIED}",
        "",
        "E",
        f"move to {CLASS_EMPTY}",
        "",
        "Q / Esc",
        "quit",
    ]

    _put_panel_lines(
        canvas,
        lines=left_lines,
        start_x=PANEL_PADDING,
        start_y=36,
        font_scale=0.62,
        line_height=34,
    )
    _put_panel_lines(
        canvas,
        lines=right_lines,
        start_x=canvas_width - SIDE_PANEL_WIDTH + PANEL_PADDING,
        start_y=36,
        font_scale=0.56,
        line_height=30,
    )
    return canvas


def _target_path(target_dir: Path, source_path: Path) -> Path:
    candidate = target_dir / source_path.name
    if not candidate.exists():
        return candidate
    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _remove_current_image(state: ReviewState) -> None:
    current_path = state.image_paths[state.index]
    current_path.unlink(missing_ok=False)
    del state.image_paths[state.index]
    if state.image_paths:
        state.index = min(state.index, len(state.image_paths) - 1)


def _move_current_image(state: ReviewState, target_dir: Path) -> None:
    current_path = state.image_paths[state.index]
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = _target_path(target_dir, current_path)
    current_path.rename(destination)
    del state.image_paths[state.index]
    if state.image_paths:
        state.index = min(state.index, len(state.image_paths) - 1)


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    working_dir = _choose_working_dir(dataset_dir, args.start_folder)
    image_paths = _collect_images(working_dir)
    if not image_paths:
        raise RuntimeError(f"No images were found in {working_dir}")

    state = ReviewState(dataset_dir=dataset_dir, working_dir=working_dir, image_paths=image_paths)
    cv2 = _require_cv2()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    left_arrow_keys = {81, 2424832}
    right_arrow_keys = {83, 2555904}
    delete_keys = {3014656, 330}

    print(
        f"Loaded working folder={working_dir} images={len(image_paths)} "
        f"dataset_dir={dataset_dir}"
    )

    while state.image_paths:
        current_path = state.image_paths[state.index]
        image = _load_image(current_path)
        scale = _fit_scale(image.shape[1], image.shape[0], args.max_width, args.max_height)
        preview = _render(image, scale=scale, state=state)
        cv2.resizeWindow(WINDOW_NAME, preview.shape[1], preview.shape[0])
        cv2.imshow(WINDOW_NAME, preview)
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        key = cv2.waitKeyEx(30)
        if key < 0:
            continue
        if key in (ord("q"), 27):
            break
        if key in left_arrow_keys:
            state.index = (state.index - 1) % len(state.image_paths)
            continue
        if key in right_arrow_keys:
            state.index = (state.index + 1) % len(state.image_paths)
            continue
        if key in delete_keys:
            _remove_current_image(state)
            continue
        if key in (ord("o"), ord("O")):
            _move_current_image(state, state.dataset_dir / CLASS_OCCUPIED)
            continue
        if key in (ord("e"), ord("E")):
            _move_current_image(state, state.dataset_dir / CLASS_EMPTY)
            continue

    cv2.destroyAllWindows()
    print(f"Review finished. Remaining in {state.working_dir.name}: {len(state.image_paths)}")


if __name__ == "__main__":
    main()
