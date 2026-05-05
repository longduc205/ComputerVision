"""Evaluation metrics: mAP, Performance Drop, Harmonic Mean."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    mAP: float
    mAP_50: float
    mAP_75: float
    mAP_per_class: Dict[int, float]
    performance_drop: Optional[float]
    harmonic_mean: Optional[float]


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_thresholds: Optional[List[float]] = None,
) -> EvaluationResult:
    """Compute COCO-style mAP metrics.

    Args:
        predictions: List of {"image_id": int, "bbox": [x,y,w,h], "score": float, "category_id": int}
        ground_truths: List of {"image_id": int, "bbox": [x,y,w,h], "category_id": int, "id": int}
        iou_thresholds: IoU thresholds for AP calculation.

    Returns:
        EvaluationResult with mAP metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = list(np.linspace(0.50, 0.95, 10))

    if not ground_truths:
        return EvaluationResult(
            mAP=0.0,
            mAP_50=0.0,
            mAP_75=0.0,
            mAP_per_class={},
            performance_drop=None,
            harmonic_mean=None,
        )

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO()
        unique_images = sorted(set(gt["image_id"] for gt in ground_truths))
        unique_cats = sorted(set(gt["category_id"] for gt in ground_truths))

        coco_gt.dataset = {
            "images": [{"id": img_id} for img_id in unique_images],
            "annotations": ground_truths,
            "categories": [{"id": cat_id} for cat_id in unique_cats],
        }
        coco_gt.createIndex()

        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = coco_gt.loadRes([])

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array(iou_thresholds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        mAP = float(stats[0])
        mAP_50 = float(stats[1])
        mAP_75 = float(stats[2])

        per_class = {}
        for cat_id in unique_cats:
            coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.params.iouThrs = np.array(iou_thresholds)
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            per_class[cat_id] = float(coco_eval_cat.stats[0])

        return EvaluationResult(
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75,
            mAP_per_class=per_class,
            performance_drop=None,
            harmonic_mean=None,
        )

    except ImportError:
        return EvaluationResult(
            mAP=0.0,
            mAP_50=0.0,
            mAP_75=0.0,
            mAP_per_class={},
            performance_drop=None,
            harmonic_mean=None,
        )


def compute_performance_drop(mAP_ID: float, mAP_OOD: float) -> float:
    """Compute Performance Drop percentage.

    PD = 100 * (mAP_ID - mAP_OOD) / mAP_ID
    Higher PD means larger generalization gap.
    """
    if mAP_ID <= 0:
        return 0.0
    return 100.0 * (mAP_ID - mAP_OOD) / mAP_ID


def compute_harmonic_mean(mAP_ID: float, mAP_OOD: float) -> float:
    """Compute Harmonic Mean of ID and OOD mAP.

    H = 2 * mAP_ID * mAP_OOD / (mAP_ID + mAP_OOD)
    Balances ID and OOD performance.
    """
    if mAP_ID + mAP_OOD <= 0:
        return 0.0
    return 2.0 * mAP_ID * mAP_OOD / (mAP_ID + mAP_OOD)


def compute_all_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    mAP_ID: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Returns:
        Dict with mAP, mAP_50, mAP_75, PD, H.
    """
    result = compute_map(predictions, ground_truths)

    metrics: Dict[str, float] = {
        "mAP": result.mAP,
        "mAP_50": result.mAP_50,
        "mAP_75": result.mAP_75,
    }

    if mAP_ID is not None and result.mAP > 0:
        metrics["PD"] = compute_performance_drop(mAP_ID, result.mAP)
        metrics["H"] = compute_harmonic_mean(mAP_ID, result.mAP)

    return metrics
