#!/usr/bin/env python3
"""Extract transparent ship PNG cutouts from a YOLO train split with SAM.

This tool reads YOLO detection labels from a training split, converts each
normalized bbox to pixel coordinates, uses the bbox as a SAM box prompt, and
writes one RGBA PNG cutout per accepted object.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TARGET_CLASSES = ("naval", "merchant", "other_vessel")
ALLOWED_TARGET_CLASSES = set(DEFAULT_TARGET_CLASSES)
METADATA_COLUMNS = [
    "cutout_file",
    "source_image",
    "source_label",
    "class_id",
    "class_name",
    "original_bbox_xyxy",
    "expanded_bbox_xyxy",
    "crop_bbox_xyxy",
    "image_width",
    "image_height",
    "bbox_area",
    "mask_area",
    "mask_bbox_area",
    "sam_score",
]


@dataclass(frozen=True)
class Candidate:
    label_index: int
    line_number: int
    class_id: int
    class_name: str
    original_bbox_xyxy: tuple[int, int, int, int]
    expanded_bbox_xyxy: tuple[int, int, int, int]
    bbox_area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Meta Segment Anything Model (SAM, not SAM2) to extract "
            "transparent ship PNG cutouts from a YOLO train split."
        )
    )
    parser.add_argument("--images-dir", required=True, type=Path, help="Path to YOLO train images.")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Path to YOLO train labels.")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help=(
            "YOLO data.yaml containing class names. If omitted or missing, "
            "the tool falls back to classes.txt next to labels-dir."
        ),
    )
    parser.add_argument("--sam-checkpoint", required=True, type=Path, help="Path to a SAM .pth checkpoint.")
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        choices=("vit_h", "vit_l", "vit_b"),
        help="SAM model type matching the checkpoint.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output cutout library directory.")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Class names to extract. Defaults to naval merchant other_vessel.",
    )
    parser.add_argument(
        "--min-bbox-area",
        type=int,
        default=100,
        help="Skip objects whose clipped pixel bbox area is smaller than this value.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=20,
        help="Skip SAM masks whose foreground pixel area is smaller than this value.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.05,
        help="Expand each bbox by this ratio before sending it to SAM as a box prompt.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save per-object preview images with bbox and mask overlays.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Device for SAM inference. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help=(
            "Optional conservative clean-up: keep only the largest connected "
            "mask component. Disabled by default to avoid losing thin ship details."
        ),
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def canonicalize_class_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def validate_train_split(images_dir: Path, labels_dir: Path) -> None:
    for name, path in (("images-dir", images_dir), ("labels-dir", labels_dir)):
        resolved = path.resolve()
        if not resolved.is_dir():
            raise ValueError(f"{name} does not exist or is not a directory: {resolved}")

        parts = {part.lower() for part in resolved.parts}
        forbidden = parts.intersection({"val", "validation", "test"})
        if forbidden:
            raise ValueError(
                f"{name} points at a non-train split ({', '.join(sorted(forbidden))}): {resolved}"
            )
        if "train" not in parts:
            raise ValueError(
                f"{name} must include a 'train' path component to prevent data leakage: {resolved}"
            )


def validate_args(args: argparse.Namespace) -> None:
    validate_train_split(args.images_dir, args.labels_dir)

    if args.min_bbox_area < 1:
        raise ValueError("--min-bbox-area must be >= 1.")
    if args.min_mask_area < 1:
        raise ValueError("--min-mask-area must be >= 1.")
    if args.bbox_margin < 0:
        raise ValueError("--bbox-margin must be >= 0.")
    if not args.sam_checkpoint.is_file():
        raise ValueError(f"SAM checkpoint does not exist: {args.sam_checkpoint.resolve()}")

    targets = {canonicalize_class_name(name) for name in args.target_classes}
    invalid = targets.difference(ALLOWED_TARGET_CLASSES)
    if invalid:
        raise ValueError(
            "Only these target classes are allowed: "
            f"{', '.join(sorted(ALLOWED_TARGET_CLASSES))}. Invalid: {', '.join(sorted(invalid))}"
        )


def load_class_names(data_yaml: Path | None, labels_dir: Path) -> dict[int, str]:
    if data_yaml and data_yaml.exists():
        if data_yaml.suffix.lower() == ".txt":
            logging.info("Loading class names from classes.txt: %s", data_yaml)
            return load_classes_txt(data_yaml)

        logging.info("Loading class names from data.yaml: %s", data_yaml)
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pyyaml. Install requirements_sam_cutout.txt.") from exc

        with data_yaml.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        names = data.get("names")
        if isinstance(names, list):
            return {index: canonicalize_class_name(str(name)) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): canonicalize_class_name(str(name)) for index, name in names.items()}
        raise ValueError(f"Unsupported or missing 'names' field in data.yaml: {data_yaml}")

    if data_yaml:
        logging.warning("data.yaml not found, falling back to classes.txt: %s", data_yaml)

    classes_txt = labels_dir.resolve().parent / "classes.txt"
    if classes_txt.is_file():
        logging.info("Loading class names from fallback classes.txt: %s", classes_txt)
        return load_classes_txt(classes_txt)

    raise FileNotFoundError(
        "Could not find class names. Provide --data-yaml or place classes.txt next to labels-dir."
    )


def load_classes_txt(path: Path) -> dict[int, str]:
    class_names: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            name = line.strip()
            if not name:
                continue
            class_names[index] = canonicalize_class_name(name)
    if not class_names:
        raise ValueError(f"No class names found in {path}")
    return class_names


def resolve_target_classes(
    class_names: dict[int, str], target_classes: Iterable[str]
) -> dict[int, str]:
    targets = {canonicalize_class_name(name) for name in target_classes}
    selected = {
        class_id: class_name
        for class_id, class_name in class_names.items()
        if class_name in targets and class_name in ALLOWED_TARGET_CLASSES
    }
    missing = targets.difference(selected.values())
    if missing:
        raise ValueError(
            "Target class names not found in dataset classes: " + ", ".join(sorted(missing))
        )
    return selected


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_bbox_to_pixel_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized xywh to clipped pixel xyxy.

    YOLO labels store the box center and size in normalized image coordinates.
    SAM expects a pixel-space xyxy box prompt, so this function scales by the
    image dimensions and clips the result to the image bounds.
    """
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    return clip_bbox(
        (
            int(max(0, min(image_width, x1))),
            int(max(0, min(image_height, y1))),
            int(max(0, min(image_width, round_up(x2)))),
            int(max(0, min(image_height, round_up(y2)))),
        ),
        image_width,
        image_height,
    )


def round_up(value: float) -> int:
    import math

    return int(math.ceil(value))


def clip_bbox(
    bbox_xyxy: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )


def expand_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    margin_ratio: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    margin_x = round_up(box_width * margin_ratio)
    margin_y = round_up(box_height * margin_ratio)
    return clip_bbox((x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y), image_width, image_height)


def bbox_area(bbox_xyxy: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def parse_label_candidates(
    label_path: Path,
    class_names: dict[int, str],
    selected_classes: dict[int, str],
    image_width: int,
    image_height: int,
    min_bbox_area: int,
    bbox_margin: float,
    skipped: Counter[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    with label_path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    if not any(line.strip() for line in lines):
        skipped["empty_label_file"] += 1
        logging.warning("Skipping empty label file: %s", label_path)
        return candidates

    for label_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            skipped["invalid_label_line"] += 1
            logging.warning("Invalid YOLO label line in %s:%d", label_path, label_index + 1)
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            skipped["invalid_label_line"] += 1
            logging.warning("Invalid YOLO values in %s:%d", label_path, label_index + 1)
            continue

        class_name = class_names.get(class_id)
        if class_name is None:
            skipped["unknown_class_id"] += 1
            continue
        if class_id not in selected_classes:
            skipped["non_target_class"] += 1
            continue

        original_bbox = yolo_bbox_to_pixel_xyxy(
            x_center, y_center, width, height, image_width, image_height
        )
        area = bbox_area(original_bbox)
        if area <= 0:
            skipped["invalid_bbox"] += 1
            continue
        if area < min_bbox_area:
            skipped["small_bbox"] += 1
            continue

        expanded_bbox = expand_bbox(original_bbox, bbox_margin, image_width, image_height)
        if bbox_area(expanded_bbox) <= 0:
            skipped["invalid_expanded_bbox"] += 1
            continue

        candidates.append(
            Candidate(
                label_index=label_index,
                line_number=label_index + 1,
                class_id=class_id,
                class_name=selected_classes[class_id],
                original_bbox_xyxy=original_bbox,
                expanded_bbox_xyxy=expanded_bbox,
                bbox_area=area,
            )
        )

    return candidates


def auto_detect_device(torch_module: Any, requested_device: str | None) -> str:
    if requested_device:
        if requested_device == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if requested_device == "mps":
            mps = getattr(torch_module.backends, "mps", None)
            if not (mps and mps.is_available()):
                raise RuntimeError("Requested --device mps, but MPS is not available.")
        return requested_device

    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps and mps.is_available():
        return "mps"
    return "cpu"


def load_sam_predictor(checkpoint: Path, model_type: str, requested_device: str | None) -> tuple[Any, str]:
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "Missing SAM dependencies. Install requirements_sam_cutout.txt and then run: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

    device = auto_detect_device(torch, requested_device)
    logging.info("Loading SAM model %s on %s", model_type, device)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam), device


def mask_to_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def keep_largest_component(mask: Any) -> Any:
    """Keep only the largest connected foreground component.

    This is optional and disabled by default because thin ship features such as
    masts may be disconnected from the hull after segmentation.
    """
    import cv2
    import numpy as np

    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 2:
        return mask

    foreground_labels = range(1, num_labels)
    largest_label = max(foreground_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    return labels == largest_label


def write_cutout_png(image_rgb: Any, mask: Any, crop_bbox: tuple[int, int, int, int], output_path: Path) -> None:
    """Write a cropped RGBA PNG where only the SAM mask remains opaque."""
    import numpy as np
    from PIL import Image

    x1, y1, x2, y2 = crop_bbox
    crop_rgb = image_rgb[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    alpha = np.where(crop_mask, 255, 0).astype(np.uint8)
    rgba = np.dstack((crop_rgb, alpha))
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def save_preview_image(
    image_rgb: Any,
    mask: Any,
    original_bbox: tuple[int, int, int, int],
    expanded_bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
    preview_path: Path,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_alpha = (mask.astype(np.uint8) * 110)
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[..., 0] = 255
    mask_rgba[..., 3] = mask_alpha
    overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, mode="RGBA"))

    preview = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(preview)
    draw_bbox(draw, original_bbox, "yellow", width=2)
    draw_bbox(draw, expanded_bbox, "deepskyblue", width=2)
    draw_bbox(draw, crop_bbox, "lime", width=2)
    preview.convert("RGB").save(preview_path)


def draw_bbox(draw: Any, bbox_xyxy: tuple[int, int, int, int], color: str, width: int = 2) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle([x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)], outline=color, width=width)


def write_metadata_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_output_dirs(output_dir: Path, target_classes: Iterable[str], save_preview: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in target_classes:
        (output_dir / canonicalize_class_name(class_name)).mkdir(parents=True, exist_ok=True)
    if save_preview:
        (output_dir / "_preview").mkdir(parents=True, exist_ok=True)


def process_dataset(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Install requirements_sam_cutout.txt.") from exc

    class_names = load_class_names(args.data_yaml, args.labels_dir)
    selected_classes = resolve_target_classes(class_names, args.target_classes)
    selected_class_names = sorted(set(selected_classes.values()))

    predictor, device = load_sam_predictor(args.sam_checkpoint, args.sam_model_type, args.device)
    ensure_output_dirs(args.output_dir, selected_class_names, args.save_preview)
    logging.info("Using class ids: %s", selected_classes)
    logging.info("Writing cutouts to: %s", args.output_dir.resolve())

    image_paths = iter_images(args.images_dir)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    extracted_per_class: Counter[str] = Counter()
    processed_images = 0
    processed_labels = 0
    images_with_labels = 0

    for image_path in tqdm(image_paths, desc="Images", unit="image"):
        processed_images += 1
        label_path = args.labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            skipped["missing_label_file"] += 1
            logging.warning("Missing label for image, skipping: %s", image_path)
            continue

        images_with_labels += 1
        processed_labels += 1

        try:
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - continue batch on corrupt images.
            skipped["image_read_failed"] += 1
            logging.warning("Could not read image %s: %s", image_path, exc)
            continue

        image_height, image_width = image_rgb.shape[:2]
        candidates = parse_label_candidates(
            label_path=label_path,
            class_names=class_names,
            selected_classes=selected_classes,
            image_width=image_width,
            image_height=image_height,
            min_bbox_area=args.min_bbox_area,
            bbox_margin=args.bbox_margin,
            skipped=skipped,
        )
        if not candidates:
            continue

        try:
            predictor.set_image(image_rgb)
        except Exception as exc:  # noqa: BLE001 - continue batch on SAM failures.
            skipped["sam_set_image_failed"] += len(candidates)
            logging.warning("SAM set_image failed for %s: %s", image_path, exc)
            continue

        for candidate in candidates:
            input_box = np.array(candidate.expanded_bbox_xyxy, dtype=np.float32)
            try:
                # SAM box prompt: the expanded pixel xyxy bbox guides SAM toward
                # the ship instance without needing any manual point prompts.
                masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
            except Exception as exc:  # noqa: BLE001 - continue batch on per-object SAM failures.
                skipped["sam_prediction_failed"] += 1
                logging.warning(
                    "SAM prediction failed for %s line %d: %s",
                    label_path,
                    candidate.line_number,
                    exc,
                )
                continue

            if masks is None or len(masks) == 0:
                skipped["sam_no_mask"] += 1
                continue

            best_index = int(np.argmax(scores))
            mask = masks[best_index].astype(bool)
            sam_score = float(scores[best_index])
            if args.keep_largest_component:
                mask = keep_largest_component(mask)

            mask_area = int(mask.sum())
            if mask_area < args.min_mask_area:
                skipped["small_mask"] += 1
                continue

            crop_bbox = mask_to_bbox(mask)
            if crop_bbox is None:
                skipped["empty_mask_bbox"] += 1
                continue

            mask_bbox_area = bbox_area(crop_bbox)
            if mask_bbox_area <= 0:
                skipped["invalid_mask_bbox"] += 1
                continue

            class_dir = args.output_dir / candidate.class_name
            cutout_name = f"{candidate.class_name}_{image_path.stem}_obj{candidate.label_index:03d}.png"
            cutout_path = class_dir / cutout_name

            try:
                write_cutout_png(image_rgb, mask, crop_bbox, cutout_path)
            except Exception as exc:  # noqa: BLE001 - continue batch on write failures.
                skipped["cutout_write_failed"] += 1
                logging.warning("Could not write cutout %s: %s", cutout_path, exc)
                continue

            if args.save_preview:
                preview_name = f"{candidate.class_name}_{image_path.stem}_obj{candidate.label_index:03d}.png"
                preview_path = args.output_dir / "_preview" / preview_name
                try:
                    save_preview_image(
                        image_rgb=image_rgb,
                        mask=mask,
                        original_bbox=candidate.original_bbox_xyxy,
                        expanded_bbox=candidate.expanded_bbox_xyxy,
                        crop_bbox=crop_bbox,
                        preview_path=preview_path,
                    )
                except Exception as exc:  # noqa: BLE001 - previews are useful but non-critical.
                    skipped["preview_write_failed"] += 1
                    logging.warning("Could not write preview %s: %s", preview_path, exc)

            extracted_per_class[candidate.class_name] += 1
            rows.append(
                {
                    "cutout_file": str(cutout_path.relative_to(args.output_dir)),
                    "source_image": str(image_path.resolve()),
                    "source_label": str(label_path.resolve()),
                    "class_id": candidate.class_id,
                    "class_name": candidate.class_name,
                    "original_bbox_xyxy": json.dumps(candidate.original_bbox_xyxy),
                    "expanded_bbox_xyxy": json.dumps(candidate.expanded_bbox_xyxy),
                    "crop_bbox_xyxy": json.dumps(crop_bbox),
                    "image_width": image_width,
                    "image_height": image_height,
                    "bbox_area": candidate.bbox_area,
                    "mask_area": mask_area,
                    "mask_bbox_area": mask_bbox_area,
                    "sam_score": f"{sam_score:.6f}",
                }
            )

    write_metadata_csv(args.output_dir, rows)
    print_summary(
        processed_images=processed_images,
        images_with_labels=images_with_labels,
        processed_labels=processed_labels,
        extracted_per_class=extracted_per_class,
        skipped=skipped,
        selected_class_names=selected_class_names,
        output_dir=args.output_dir,
        device=device,
    )
    return 0


def print_summary(
    processed_images: int,
    images_with_labels: int,
    processed_labels: int,
    extracted_per_class: Counter[str],
    skipped: Counter[str],
    selected_class_names: Iterable[str],
    output_dir: Path,
    device: str,
) -> None:
    print("\nSummary")
    print(f"  device: {device}")
    print(f"  scanned images: {processed_images}")
    print(f"  images with labels: {images_with_labels}")
    print(f"  processed labels: {processed_labels}")
    print("  extracted cutouts per class:")
    for class_name in selected_class_names:
        print(f"    {class_name}: {extracted_per_class.get(class_name, 0)}")
    print("  skipped objects/files:")
    if skipped:
        for reason, count in sorted(skipped.items()):
            print(f"    {reason}: {count}")
    else:
        print("    none: 0")
    print(f"  metadata: {output_dir / 'metadata.csv'}")


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        validate_args(args)
        return process_dataset(args)
    except Exception as exc:  # noqa: BLE001 - provide a concise CLI error.
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
