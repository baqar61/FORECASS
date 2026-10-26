import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PRED_ROOT = r""
GT_ROOT = r""
TARGET_SIZE = (640, 320)

CLASS_NAMES = ["flat", "construction", "object", "nature", "sky", "human", "vehicle"]
PALETTE = np.array([
    [128, 64, 128], [70, 70, 70], [153, 153, 153], [107, 142, 35],
    [70, 130, 180], [220, 20, 60], [0, 0, 142],
], dtype=np.uint8)
OBJECT_COLORS = [(232, 174, 0), (255, 255, 0)]


def core_id(stem):
    for suffix in ("_gtfine_labeltrainids", "_gtfine_labeltrainid", "_gtfine_labelids",
                   "_gtfine_labelid", "_leftimg8bit"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def collect_masks(root):
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = {}
    by_name = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative = path.relative_to(root)
        name = core_id(relative.stem)
        key = (relative.parent / name).as_posix()
        if key in files:
            raise ValueError(f"Duplicate mask ID: {key} in {root}")
        files[key] = path
        by_name[name].append(path)
    if not files:
        raise ValueError(f"No PNG masks found in {root}")
    return files, by_name


def build_pairs(pred_root, gt_root):
    predictions, pred_names = collect_masks(pred_root)
    targets, gt_names = collect_masks(gt_root)
    pairs, missing, used = [], [], set()
    for key, target in targets.items():
        prediction = predictions.get(key)
        if prediction is None:
            name = core_id(target.stem)
            candidates = pred_names.get(name, [])
            if len(candidates) == 1 and len(gt_names[name]) == 1:
                prediction = candidates[0]
            elif candidates:
                raise ValueError(f"Ambiguous filename {name}; preserve matching subfolders")
        if prediction is None:
            missing.append(str(target))
            continue
        if prediction in used:
            raise ValueError(f"Prediction matched more than once: {prediction}")
        used.add(prediction)
        pairs.append((prediction, target))
    if missing:
        examples = "\n".join(missing[:5])
        raise ValueError(f"Missing predictions for {len(missing)} ground-truth masks:\n{examples}")
    return pairs, len(predictions) - len(used)


def rgb_to_ids(rgb, path, ground_truth=False):
    ids = np.full(rgb.shape[:2], 255, dtype=np.int64)
    known = np.all(rgb == 0, axis=-1)
    for class_id, color in enumerate(PALETTE):
        selected = np.all(rgb == color, axis=-1)
        ids[selected] = class_id
        known |= selected
    for color in OBJECT_COLORS:
        selected = np.all(rgb == color, axis=-1)
        ids[selected] = 2
        known |= selected
    if ground_truth and not known.all():
        colors = np.unique(rgb[~known].reshape(-1, 3), axis=0)[:5].tolist()
        raise ValueError(f"Unrecognized ground-truth colors in {path}: {colors}; use shared-class ID masks")
    return ids


def load_mask(path, size, ground_truth=False):
    with Image.open(path) as source:
        source = source.resize(size, Image.Resampling.NEAREST)
        if source.mode in ("RGB", "RGBA"):
            rgb = np.array(source.convert("RGB"))
            if source.mode == "RGBA":
                rgb[np.array(source)[..., 3] == 0] = 0
            ids = rgb_to_ids(rgb, path, ground_truth)
            if source.mode == "RGBA":
                ids[np.array(source)[..., 3] == 0] = 255
        elif source.mode in ("L", "P", "I", "I;16"):
            ids = np.array(source, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported mask mode {source.mode} in {path}")
    if ground_truth:
        values = set(np.unique(ids).tolist())
        if not values.issubset(set(range(7)) | {7, 255}):
            raise ValueError(f"{path}: labels must already be mapped to 0..6, with 7/255 ignored; found {sorted(values)}")
    return ids


def confusion_matrix(target, prediction):
    valid = (target >= 0) & (target < 7)
    bad = valid & ((prediction < 0) | (prediction >= 7))
    if bad.any():
        values = np.unique(prediction[bad]).tolist()
        raise ValueError(f"Invalid prediction IDs {values} on {int(bad.sum())} evaluated pixels; check the prediction palette")
    return np.bincount(7 * target[valid] + prediction[valid], minlength=49).reshape(7, 7)


def evaluate(pred_root, gt_root, size=TARGET_SIZE):
    pairs, extra = build_pairs(pred_root, gt_root)
    print(f"Matched images: {len(pairs)}; unused predictions: {extra}")
    hist = np.zeros((7, 7), dtype=np.int64)
    for prediction_path, target_path in tqdm(pairs, desc="Evaluating", unit="image"):
        prediction = load_mask(prediction_path, size)
        target = load_mask(target_path, size, ground_truth=True)
        try:
            hist += confusion_matrix(target, prediction)
        except ValueError as error:
            raise ValueError(f"{prediction_path}: {error}") from error
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.full(7, np.nan), where=union > 0)
    if not np.isfinite(iou).any():
        raise ValueError("No valid ground-truth pixels were evaluated")
    mean_iou = float(np.nanmean(iou) * 100)
    print("\nPer-class IoU:")
    for name, score in zip(CLASS_NAMES, iou):
        value = f"{score * 100:.2f}%" if np.isfinite(score) else "N/A"
        print(f"{name:<13} {value}")
    print(f"mIoU: {mean_iou:.2f}%")
    return {
        "images": len(pairs), "unused_predictions": extra, "size": list(size),
        "miou": mean_iou,
        "class_iou": {name: float(score * 100) if np.isfinite(score) else None
                      for name, score in zip(CLASS_NAMES, iou)},
        "confusion_matrix": hist.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved FORECASS segmentation masks")
    parser.add_argument("--pred-root", default=PRED_ROOT)
    parser.add_argument("--gt-root", default=GT_ROOT)
    parser.add_argument("--size", nargs=2, type=int, default=TARGET_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, help="Optional JSON result file")
    args = parser.parse_args()
    if not args.pred_root or not args.gt_root:
        parser.error("Set PRED_ROOT and GT_ROOT at the top, or pass --pred-root and --gt-root")
    if any(value < 1 for value in args.size):
        parser.error("Image dimensions must be positive")
    result = evaluate(args.pred_root, args.gt_root, tuple(args.size))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
