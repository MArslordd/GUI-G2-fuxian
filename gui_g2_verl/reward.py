"""GUI-G2 Gaussian reward for verl GRPO.

The reward follows the paper and the official GUI-G2 implementation:
- Gaussian point reward: predicted center under the target Gaussian.
- Gaussian coverage reward: Bhattacharyya coefficient between predicted and
  target Gaussians.
- Format reward: model must emit a bbox like [x1,y1,x2,y2].
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def _coerce_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            match = BBOX_RE.search(value)
            if not match:
                return None
            return [float(match.group(i)) for i in range(1, 5)]
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in value for k in keys):
            return [float(value[k]) for k in keys]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(x) for x in value]
    return None


def _extract_pred_bbox(solution_str: str) -> list[float] | None:
    match = BBOX_RE.search(solution_str)
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def _valid_bbox(box: list[float], eps: float = 1e-6) -> bool:
    return len(box) == 4 and box[2] > box[0] + eps and box[3] > box[1] + eps


def _normalize_if_needed(box: list[float], width: float | None, height: float | None) -> list[float]:
    if width and height and max(box) <= 1.5:
        return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    return box


def _gaussian_point(pred: list[float], gt: list[float], alpha: float, eps: float) -> float:
    pcx = (pred[0] + pred[2]) / 2.0
    pcy = (pred[1] + pred[3]) / 2.0
    gcx = (gt[0] + gt[2]) / 2.0
    gcy = (gt[1] + gt[3]) / 2.0
    sigma_x = max(alpha * (gt[2] - gt[0]), eps)
    sigma_y = max(alpha * (gt[3] - gt[1]), eps)
    exponent = -0.5 * (((pcx - gcx) ** 2) / (sigma_x**2) + ((pcy - gcy) ** 2) / (sigma_y**2))
    return math.exp(exponent)


def _gaussian_coverage(pred: list[float], gt: list[float], alpha: float, eps: float) -> float:
    pcx = (pred[0] + pred[2]) / 2.0
    pcy = (pred[1] + pred[3]) / 2.0
    gcx = (gt[0] + gt[2]) / 2.0
    gcy = (gt[1] + gt[3]) / 2.0

    psx = max(alpha * (pred[2] - pred[0]), eps)
    psy = max(alpha * (pred[3] - pred[1]), eps)
    gsx = max(alpha * (gt[2] - gt[0]), eps)
    gsy = max(alpha * (gt[3] - gt[1]), eps)

    avg_x = (psx**2 + gsx**2) / 2.0
    avg_y = (psy**2 + gsy**2) / 2.0
    term1 = 0.125 * (((pcx - gcx) ** 2) / avg_x + ((pcy - gcy) ** 2) / avg_y)
    det_avg = avg_x * avg_y
    det_pred = (psx**2) * (psy**2)
    det_gt = (gsx**2) * (gsy**2)
    term2 = 0.5 * math.log((det_avg + eps) / math.sqrt(det_pred * det_gt + eps))
    return math.exp(-(term1 + term2))


def _format_reward(solution_str: str) -> float:
    return 1.0 if BBOX_RE.fullmatch(solution_str.strip()) else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    point_weight: float = 0.45,
    coverage_weight: float = 0.45,
    format_weight: float = 0.10,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Return a verl-compatible reward dict."""

    del data_source
    extra_info = extra_info or {}

    if isinstance(ground_truth, dict):
        gt_box = _coerce_bbox(ground_truth.get("bbox") or ground_truth.get("box") or ground_truth.get("abs_box"))
        width = ground_truth.get("width") or ground_truth.get("image_width") or extra_info.get("width")
        height = ground_truth.get("height") or ground_truth.get("image_height") or extra_info.get("height")
    else:
        gt_box = _coerce_bbox(ground_truth)
        width = extra_info.get("width")
        height = extra_info.get("height")

    pred_box = _extract_pred_bbox(solution_str)
    fmt = _format_reward(solution_str)
    if pred_box is None or gt_box is None:
        return {"score": 0.0, "point": 0.0, "coverage": 0.0, "format": fmt}

    pred_box = _normalize_if_needed(pred_box, width, height)
    gt_box = _normalize_if_needed(gt_box, width, height)

    if not _valid_bbox(pred_box) or not _valid_bbox(gt_box):
        return {"score": format_weight * fmt, "point": 0.0, "coverage": 0.0, "format": fmt}

    point = _gaussian_point(pred_box, gt_box, alpha=alpha, eps=eps)
    coverage = _gaussian_coverage(pred_box, gt_box, alpha=alpha, eps=eps)
    score = point_weight * point + coverage_weight * coverage + format_weight * fmt
    return {
        "score": float(round(score, 6)),
        "point": float(round(point, 6)),
        "coverage": float(round(coverage, 6)),
        "format": float(fmt),
    }
