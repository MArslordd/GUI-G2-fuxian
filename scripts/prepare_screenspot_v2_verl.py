"""Prepare a small ScreenSpot-v2-variants split for verl multimodal GRPO."""

from __future__ import annotations

import argparse
import io
import os
from typing import Any

from PIL import Image


PROMPT_TEMPLATE = (
    "<image>\n"
    "Outline the position corresponding to the instruction: {instruction}. "
    "The output should be only [x1,y1,x2,y2]."
)


def _first_key(example: dict[str, Any], candidates: list[str]) -> str | None:
    return next((key for key in candidates if key in example and example[key] is not None), None)


def _bbox_from_example(example: dict[str, Any], bbox_field: str | None) -> list[float]:
    key = bbox_field or _first_key(
        example,
        ["bbox", "box", "abs_box", "gt_bbox", "target_bbox", "element_bbox", "rect"],
    )
    if key is None:
        raise KeyError(f"No bbox field found. Available fields: {sorted(example.keys())}")
    value = example[key]
    if isinstance(value, dict):
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return [float(value[k]) for k in ("x1", "y1", "x2", "y2")]
        if all(k in value for k in ("left", "top", "right", "bottom")):
            return [float(value[k]) for k in ("left", "top", "right", "bottom")]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(x) for x in value]
    raise ValueError(f"Unsupported bbox value in field {key!r}: {value!r}")


def _image_to_bytes_feature(image: Any) -> tuple[dict[str, bytes | None], int, int]:
    if isinstance(image, dict) and image.get("bytes"):
        raw = image["bytes"]
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
        return {"bytes": raw, "path": None}, width, height
    if isinstance(image, dict) and image.get("path"):
        image = image["path"]
    if isinstance(image, str):
        with Image.open(image) as img:
            pil = img.convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image value: {type(image)}")

    width, height = pil.size
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}, width, height


def _to_verl_row(example: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    instruction_key = args.instruction_field or _first_key(
        example,
        ["instruction", "action", "description", "query", "question", "target"],
    )
    image_key = args.image_field or _first_key(example, ["image", "img", "screenshot"])
    if instruction_key is None or image_key is None:
        raise KeyError(f"Need instruction and image fields. Available fields: {sorted(example.keys())}")

    instruction = str(example[instruction_key])
    image_feature, width, height = _image_to_bytes_feature(example[image_key])
    bbox = _bbox_from_example(example, args.bbox_field)
    if max(bbox) <= 1.5:
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    bbox = [round(float(x), 3) for x in bbox]

    return {
        "data_source": args.dataset,
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(instruction=instruction)}],
        "images": [image_feature],
        "ability": "gui_grounding",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"bbox": bbox, "width": width, "height": height},
        },
        "extra_info": {
            "split": args.split,
            "index": idx,
            "instruction": instruction,
            "width": width,
            "height": height,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="likaixin/ScreenSpot-v2-variants")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/screenspot_v2_verl")
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--bbox-field", default=None)
    args = parser.parse_args()

    from datasets import Dataset, load_dataset

    os.makedirs(args.output_dir, exist_ok=True)
    raw = load_dataset(args.dataset, split=args.split)
    raw = raw.shuffle(seed=args.seed)
    total = min(len(raw), args.max_train_samples + args.max_val_samples)
    raw = raw.select(range(total))

    rows = [_to_verl_row(example, idx, args) for idx, example in enumerate(raw)]
    train_rows = rows[: args.max_train_samples]
    val_rows = rows[args.max_train_samples :]
    if not val_rows:
        val_rows = train_rows[: min(4, len(train_rows))]

    Dataset.from_list(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(os.path.join(args.output_dir, "val.parquet"))
    print(f"Wrote {len(train_rows)} train and {len(val_rows)} val samples to {args.output_dir}")


if __name__ == "__main__":
    main()
