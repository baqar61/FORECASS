import os
import glob
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

VALID_CLASSES = list(range(7)) 
NUM_CLASSES = len(VALID_CLASSES)
PRED_ROOT = r"C:\PHD_online_learning\ADVENT\results_idd"
GT_ROOT   = r"C:\PHD_online_learning\DeepLabV3Plus-Pytorch-master_lora\datasets\data\cityscapesidd\gtFine\val"
TARGET_SIZE = (640, 320)  
CLASS_NAMES = [
    "flat",         
    "construction", 
    "object",      
    "nature",    
    "sky",          
    "human",     
    "vehicle",      
]
PALETTE = np.array([
    [128,  64, 128],  
    [ 70,  70,  70],  
    [153, 153, 153],  
    [107, 142,  35], 
    [ 70, 130, 180],  
    [220,  20,  60],  
    [  0,   0, 142],  
], dtype=np.uint8)
def has_subfolders(root):
    for x in os.listdir(root):
        if os.path.isdir(os.path.join(root, x)):
            return True
    return False
def collect_files(root, pattern="*.png"):
    root_has_sub = has_subfolders(root)
    index = {}
    for f in glob.glob(os.path.join(root, "**", pattern), recursive=True):
        rel = os.path.relpath(f, root)
        is_direct = (os.sep not in rel)

        if is_direct and root_has_sub:
            continue
        base = os.path.basename(f)
        stem = os.path.splitext(base)[0]
        core = normalize_core_id(stem)
        index.setdefault(core, []).append(f)
    return index

def normalize_core_id(stem):
    s_low = stem.lower()

    suffixes = [
        "_leftimg8bit",
        "_gtfine_labelids",
        "_gtfine_labeltrainids",
        "_gtfine_labeltrainid",   
        "_gtfine_labelid",        
    ]

    for suf in suffixes:
        if s_low.endswith(suf):
            cut = len(stem) - len(suf)
            return stem[:cut]
    return stem

def rgb_mask_to_ids(rgb_mask):

    h, w, _ = rgb_mask.shape
    out = np.full((h, w), 255, dtype=np.int64)
    for cls_id, color in enumerate(PALETTE):
        match = np.all(rgb_mask == color.reshape(1, 1, 3), axis=2)
        out[match] = cls_id

    return out


def load_pred_mask(path):

    img = np.array(Image.open(path))

    if img.ndim == 2:
        pred_ids = img.astype(np.int64)
    elif img.ndim == 3 and img.shape[2] == 3:
        pred_ids = rgb_mask_to_ids(img.astype(np.uint8))
    else:
        raise ValueError(f"Unexpected pred shape {img.shape} at {path}")
    pred_mask_img = Image.fromarray(pred_ids.astype(np.uint8), mode="L")
    pred_mask_resized = pred_mask_img.resize(
        TARGET_SIZE,
        resample=Image.NEAREST
    )
    pred_mask_resized = np.array(pred_mask_resized, dtype=np.int64)
    return pred_mask_resized


def load_gt_mask(path):

    gt_img = Image.open(path)
    gt_img = gt_img.resize(
        TARGET_SIZE,
        resample=Image.NEAREST
    )
    gt_mask = np.array(gt_img, dtype=np.int64)
    return gt_mask


def build_pair_list(pred_root, gt_root):

    pred_index = collect_files(pred_root)
    gt_index   = collect_files(gt_root)
    pairs = []
    for core_id, gt_list in gt_index.items():
        if core_id in pred_index:
            pairs.append((pred_index[core_id][0], gt_list[0]))
    return pairs

def fast_confusion_matrix(gt, pred, num_classes, ignore_mask):

    mask = (~ignore_mask) & (gt >= 0) & (gt < num_classes)
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    pred_valid = np.clip(pred_valid, 0, num_classes - 1)
    hist = np.bincount(
        gt_valid * num_classes + pred_valid,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    return hist

def compute_iou(conf_mat):

    tp = np.diag(conf_mat).astype(np.float64)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.zeros_like(tp, dtype=np.float64)
    valid = denom > 0
    iou[valid] = tp[valid] / denom[valid]
    return iou

def main():
    pairs = build_pair_list(PRED_ROOT, GT_ROOT)
    if len(pairs) == 0:
        print("No matching files found")
        return
    conf_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for pred_path, gt_path in tqdm(pairs, desc="Evaluating", unit="img"):
        pred_mask = load_pred_mask(pred_path)
        gt_mask   = load_gt_mask(gt_path)
        ignore_mask = ~np.isin(gt_mask, VALID_CLASSES)

        conf_total += fast_confusion_matrix(
            gt=gt_mask,
            pred=pred_mask,
            num_classes=NUM_CLASSES,
            ignore_mask=ignore_mask
        )
    iou_per_class = compute_iou(conf_total)
    miou = float(np.mean(iou_per_class)) * 100.0

    print("Per-class IoU (0..6):")
    for cid, val in enumerate(iou_per_class):
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        print(f"{cid:>2} {cname:<13}: {val * 100.0:.2f}%")

    print(f"mIoU Total: {miou:.2f}%")

if __name__ == "__main__":
    main()
