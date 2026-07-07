"""Prepare a small ScreenSpot-v2-variants split for verl multimodal GRPO."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
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
        ["bbox", "box", "abs_box", "gt_bbox", "target_bbox", "element_bbox", "rect", "bbox_xyxy"],
    )
    if key is None:
        raise KeyError(f"No bbox field found. Available fields: {sorted(example.keys())}")
    value = example[key]
    if isinstance(value, dict):
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return [float(value[k]) for k in ("x1", "y1", "x2", "y2")]
        if all(k in value for k in ("left", "top", "right", "bottom")):
            return [float(value[k]) for k in ("left", "top", "right", "bottom")]
        if all(k in value for k in ("x", "y", "width", "height")):
            x1 = float(value["x"])
            y1 = float(value["y"])
            return [x1, y1, x1 + float(value["width"]), y1 + float(value["height"])]
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


def _looks_like_instruction_key(key: str) -> bool:
    key_lower = key.lower()
    return any(token in key_lower for token in ["instruction", "action", "description", "query", "question", "target"])


def _choose_instruction_key(example: dict[str, Any], requested: str | None) -> str | None:
    if requested and requested in example and example[requested] is not None:
        return requested
    if requested:
        print(f"Warning: requested instruction field {requested!r} was not found; falling back to auto-detection.")
    return _first_key(
        example,
        [
            "instruction",
            "action",
            "description",
            "original_instruction",
            "short_instruction",
            "query",
            "question",
            "target",
            "text",
            "label",
        ],
    ) or next((key for key in example if _looks_like_instruction_key(key) and example[key] is not None), None)


def _choose_image_key(example: dict[str, Any], requested: str | None) -> str | None:
    if requested and requested in example and example[requested] is not None:
        return requested
    if requested:
        print(f"Warning: requested image field {requested!r} was not found; falling back to auto-detection.")
    return _first_key(
        example,
        ["image", "img", "screenshot", "image_path", "img_path", "img_filename", "file_name", "filename", "image_filename"],
    )


def _read_annotation_file(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "annotations", "examples", "items", "samples"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported annotation structure in {path}")


def _find_image_repo_path(example: dict[str, Any], image_files: dict[str, str]) -> str | None:
    image_value = None
    for key in ["image", "img", "screenshot", "image_path", "img_path", "img_filename", "file_name", "filename", "image_filename"]:
        if key in example and isinstance(example[key], str):
            image_value = example[key]
            break
    if image_value is None:
        return None
    normalized = image_value.replace("\\", "/")
    basename = os.path.basename(normalized)
    if normalized in image_files or basename in image_files:
        return image_files.get(normalized) or image_files.get(basename)
    return next((repo_path for repo_path in image_files.values() if repo_path.endswith(normalized)), None)


def _load_examples_from_hf_annotations(args: argparse.Namespace, total: int) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_files = list_repo_files(args.dataset, repo_type="dataset")
    annotation_files = [
        path
        for path in repo_files
        if path.startswith("annotations/") and path.lower().endswith((".json", ".jsonl"))
    ]
    if not annotation_files:
        raise RuntimeError(f"No annotations/*.json or annotations/*.jsonl files found in {args.dataset}.")

    image_files = {
        os.path.basename(path): path
        for path in repo_files
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    }
    image_files.update(
        {
            path: path
            for path in repo_files
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        }
    )

    examples: list[dict[str, Any]] = []
    for annotation_file in annotation_files:
        local_annotation = hf_hub_download(args.dataset, annotation_file, repo_type="dataset")
        for example in _read_annotation_file(local_annotation):
            if isinstance(example, dict):
                examples.append(example)

    random.Random(args.seed).shuffle(examples)
    selected: list[dict[str, Any]] = []
    for example in examples:
        repo_image_path = _find_image_repo_path(example, image_files)
        if repo_image_path is None:
            continue
        local_image = hf_hub_download(args.dataset, repo_image_path, repo_type="dataset")
        example = dict(example)
        example["image"] = local_image
        selected.append(example)
        if len(selected) >= total:
            break

    if not selected:
        sample_keys = sorted(examples[0].keys()) if examples else []
        raise RuntimeError(
            "Could not pair annotations with image files. "
            f"Sample annotation keys: {sample_keys}. "
            "Pass --image-field if the dataset uses a custom image field."
        )
    return selected


def _load_examples(args: argparse.Namespace, total: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    raw = load_dataset(args.dataset, split=args.split)
    raw = raw.shuffle(seed=args.seed)
    raw = raw.select(range(min(len(raw), total)))
    examples = [dict(example) for example in raw]
    if not examples:
        return []

    first = examples[0]
    has_instruction = _choose_instruction_key(first, args.instruction_field) is not None
    has_bbox = args.bbox_field in first if args.bbox_field else _first_key(
        first,
        ["bbox", "box", "abs_box", "gt_bbox", "target_bbox", "element_bbox", "rect", "bbox_xyxy"],
    ) is not None
    if has_instruction and has_bbox:
        return examples

    print(
        "The loaded dataset split does not expose instruction/bbox columns. "
        "Falling back to Hugging Face repository annotations."
    )
    return _load_examples_from_hf_annotations(args, total)


def _to_verl_row(example: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    instruction_key = _choose_instruction_key(example, args.instruction_field)
    image_key = _choose_image_key(example, args.image_field)
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
    parser.add_argument("--instruction-field", default=None)
    parser.add_argument("--image-field", default=None)
    parser.add_argument("--bbox-field", default=None)
    args = parser.parse_args()

    from datasets import Dataset

    os.makedirs(args.output_dir, exist_ok=True)
    total = args.max_train_samples + args.max_val_samples
    examples = _load_examples(args, total)

    rows = [_to_verl_row(example, idx, args) for idx, example in enumerate(examples)]
    train_rows = rows[: args.max_train_samples]
    val_rows = rows[args.max_train_samples :]
    if not val_rows:
        val_rows = train_rows[: min(4, len(train_rows))]

    Dataset.from_list(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(os.path.join(args.output_dir, "val.parquet"))
    print(f"Wrote {len(train_rows)} train and {len(val_rows)} val samples to {args.output_dir}")


if __name__ == "__main__":
    main()
