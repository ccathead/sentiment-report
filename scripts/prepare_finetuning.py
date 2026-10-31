"""Prepare separate MVSA review candidates and export reviewed LLaMA-Factory data."""
import argparse
import html
import json
import random
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentiment_lab.core import LABELS, file_hash, read_json, read_jsonl, render_prompt, within, write_json


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def exclusions(frozen):
    rows = read_jsonl(frozen / "annotations.jsonl") + read_jsonl(frozen / "excluded.jsonl")
    return ({str(r["source_id"]) for r in rows}, {r["image_sha256"] for r in rows},
            {text_key(r["text"]) for r in rows})


def check_rows(rows, root, frozen):
    blocked_ids, blocked_images, blocked_texts = exclusions(frozen)
    seen_ids, seen_images, seen_texts = set(), set(), set()
    for row in rows:
        identifier, image, text = str(row["source_id"]), row["image_sha256"], text_key(row["text"])
        if not text or "<image>" in row["text"]:
            raise ValueError("Empty text or reserved image token")
        if (identifier in blocked_ids | seen_ids or image in blocked_images | seen_images
                or text in blocked_texts | seen_texts):
            raise ValueError(f"Overlap/duplicate: {row['id']}")
        if row["id"] != f"mvsa_{identifier}" or file_hash(within(root, row["image"])) != image:
            raise ValueError(f"Identity/image integrity mismatch: {row['id']}")
        seen_ids.add(identifier)
        seen_images.add(image)
        seen_texts.add(text)


def prepare(source, frozen, out, per_class=120, seed=20261031):
    if out.exists():
        raise FileExistsError(out)
    blocked_ids, blocked_images, blocked_texts = exclusions(frozen)
    groups = {label: [] for label in LABELS}
    decoding_skipped = []
    # Agreement is a sampling stratum only, never a joint gold label.
    for line in (source / "labelResultAll.txt").read_text(encoding="utf-8-sig").splitlines()[1:]:
        identifier, labels = line.split()
        text_label, image_label = labels.split(",")
        if text_label != image_label or text_label not in groups or identifier in blocked_ids:
            continue
        image, text_file = source / "data" / f"{identifier}.jpg", source / "data" / f"{identifier}.txt"
        if not image.is_file() or not text_file.is_file():
            continue
        # The MVSA text files include the source post; preserve raw files separately.
        try:
            text = " ".join(html.unescape(text_file.read_text(encoding="utf-8-sig")).split())
        except UnicodeDecodeError:
            decoding_skipped.append(identifier)
            continue
        image_hash, normalised = file_hash(image), text_key(text)
        if not normalised or "<image>" in text or image_hash in blocked_images or normalised in blocked_texts:
            continue
        groups[text_label].append((identifier, image, text_file, text, image_hash, normalised))
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        rng.shuffle(groups[label])
        kept = 0
        for identifier, image, text_file, text, image_hash, normalised in groups[label]:
            if image_hash in blocked_images or normalised in blocked_texts:
                continue
            selected.append((label, identifier, image, text_file, text, image_hash))
            blocked_images.add(image_hash)
            blocked_texts.add(normalised)
            kept += 1
            if kept == per_class:
                break
        if kept < per_class:
            raise ValueError(f"Not enough distinct {label} candidates: {kept}/{per_class}")
    rng.shuffle(selected)
    (out / "images").mkdir(parents=True)
    (out / "source_text").mkdir()
    rows = []
    for label, identifier, image, text_file, text, image_hash in selected:
        row = {"id": f"mvsa_{identifier}", "source_id": identifier, "split": "candidate",
               "image": f"images/mvsa_{identifier}.jpg", "text": text, "image_sha256": image_hash,
               "source_text_label": label, "source_image_label": label,
               "human_final_label": None, "human_review_status": "pending",
               "ai_proposed_joint_label": None, "ai_confidence": "未生成",
               "visible_evidence_zh": "待人工检查", "annotation_rationale_zh": "来源单模态一致不代表联合标签已确认"}
        shutil.copy2(image, out / row["image"])
        shutil.copy2(text_file, out / "source_text" / text_file.name)
        rows.append(row)
    check_rows(rows, out, frozen)
    (out / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    metadata = {"dataset_name": "mvsa_finetuning_candidates", "version": "2026-10-31",
                "annotations_sha256": file_hash(out / "annotations.jsonl"), "seed": seed,
                "candidate_count": len(rows), "source_agreement_strata": dict(Counter(r["source_text_label"] for r in rows)),
                "frozen_manifest_sha256": file_hash(frozen / "manifest.sha256.json"),
                "source_labels_sha256": file_hash(source / "labelResultAll.txt"),
                "non_utf8_skipped_ids": decoding_skipped,
                "near_duplicate_review": "pending; exact ID/image hash/normalised text checked",
                "training_ready": False}
    write_json(out / "metadata.json", metadata)
    payload = {**metadata, "dataset_version": metadata["version"], "annotations": rows}
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "templates/review_template.html").read_text(encoding="utf-8")
    (out / "review.html").write_text(template.replace("__DATA__", encoded), encoding="utf-8")
    shutil.copy2(ROOT / "docs/annotation-guide.md", out / "ANNOTATION_GUIDE.md")
    return metadata


def export(packet, reviews, frozen, out, val_fraction=1 / 6, seed=20261031,
           *, label_policy="human-reviewed", source_labels=None):
    if out.exists():
        raise FileExistsError(out)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if label_policy not in ("human-reviewed", "source-agreement"):
        raise ValueError("Unknown label policy")
    metadata = read_json(packet / "metadata.json")
    actual_hash = file_hash(packet / "annotations.jsonl")
    if metadata["annotations_sha256"] != actual_hash:
        raise ValueError("Review/annotation version mismatch")
    if metadata["frozen_manifest_sha256"] != file_hash(frozen / "manifest.sha256.json"):
        raise ValueError("Frozen evaluation dataset changed")
    rows = read_jsonl(packet / "annotations.jsonl")
    check_rows(rows, packet, frozen)
    if label_policy == "source-agreement":
        if reviews is not None:
            raise ValueError("Source policy must not masquerade as human review")
        if source_labels is None or file_hash(source_labels) != metadata["source_labels_sha256"]:
            raise ValueError("Original source label file/hash required")
        source_map = {}
        for line in Path(source_labels).read_text(encoding="utf-8-sig").splitlines()[1:]:
            identifier, pair = line.split()
            if identifier in source_map:
                raise ValueError("Duplicate source label ID")
            source_map[identifier] = pair.split(",")
        by_id = {}
        for row in rows:
            pair = source_map.get(str(row["source_id"]))
            if (pair is None or len(pair) != 2 or pair[0] != pair[1] or pair[0] not in LABELS
                    or pair != [row["source_text_label"], row["source_image_label"]]):
                raise ValueError(f"Source label agreement mismatch: {row['id']}")
            by_id[row["id"]] = {"label": pair[0], "label_provenance": "MVSA-Single text/image label agreement",
                                "human_review_status": "not_reviewed", "human_reviewed": False}
    else:
        if reviews is None:
            raise ValueError("Human policy requires a review file")
        review = read_json(reviews)
        if review.get("annotations_sha256") != actual_hash:
            raise ValueError("Review/annotation version mismatch")
        entries = review.get("reviews", [])
        by_id = {r["id"]: r for r in entries}
        if len(by_id) != len(entries) or set(by_id) != {r["id"] for r in rows}:
            raise ValueError("Every candidate needs one review, including exclusions")
    groups = {label: [] for label in LABELS}
    excluded = []
    for row in rows:
        r = by_id[row["id"]]
        if label_policy == "human-reviewed" and (not r.get("reviewer", "").strip() or not r.get("reviewed_at")):
            raise ValueError("Reviewer and review time required")
        if r["label"] == "exclude":
            if not r.get("notes", "").strip():
                raise ValueError("Exclusion requires a reason")
            excluded.append(row["id"])
        elif r["label"] in LABELS:
            groups[r["label"]].append(row)
        else:
            raise ValueError("Invalid reviewed label")
    rng = random.Random(seed)
    partitions = {"train": [], "val": []}
    for label, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"Need at least two reviewed samples of {label}")
        group.sort(key=lambda r: r["id"])
        rng.shuffle(group)
        nval = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        partitions["val"].extend(group[:nval])
        partitions["train"].extend(group[nval:])
    (out / "images").mkdir(parents=True)
    template = (ROOT / "prompts/baseline.txt").read_text(encoding="utf-8")
    inventory = []
    for split, group in partitions.items():
        data = []
        for row in sorted(group, key=lambda r: r["id"]):
            shutil.copy2(within(packet, row["image"]), out / row["image"])
            data.append({"instruction": "<image>\n" + render_prompt(template, row["text"], "multimodal"),
                         "input": "", "output": by_id[row["id"]]["label"], "images": [row["image"]]})
            inventory.append({**row, "split": split, "training_label": by_id[row["id"]]["label"],
                              **({"review": by_id[row["id"]]} if label_policy == "human-reviewed" else
                                 {"label_assignment": by_id[row["id"]], "human_review_status": "not_reviewed"})})
        write_json(out / f"{split}.json", data)
    write_json(out / "dataset_info.json", {
        f"sentiment_{split}": {"file_name": f"{split}.json", "formatting": "alpaca",
                              "columns": {"prompt": "instruction", "query": "input", "response": "output", "images": "images"}}
        for split in partitions})
    write_json(out / "split_manifest.json", {"seed": seed, "val_fraction": val_fraction,
               "label_policy": label_policy, "human_joint_reviewed": label_policy == "human-reviewed",
               "source_labels_sha256": file_hash(source_labels) if source_labels else None,
               "review_sha256": file_hash(reviews) if reviews else None,
               "export_script_sha256": file_hash(Path(__file__)),
               "prompt_sha256": file_hash(ROOT / "prompts/baseline.txt"),
               "candidate_annotations_sha256": actual_hash, "frozen_manifest_sha256": metadata["frozen_manifest_sha256"],
               "counts": {s: dict(Counter(by_id[r["id"]]["label"] for r in g)) for s, g in partitions.items()},
               "excluded": excluded, "samples": inventory,
               "files": {str(p.relative_to(out)).replace("\\", "/"): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return {s: len(g) for s, g in partitions.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--per-class", type=int, default=120)
    exp = sub.add_parser("export")
    exp.add_argument("--packet", type=Path, required=True)
    exp.add_argument("--reviews", type=Path)
    exp.add_argument("--label-policy", choices=("human-reviewed", "source-agreement"), default="human-reviewed")
    exp.add_argument("--source-labels", type=Path)
    exp.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.action == "prepare":
        if args.per_class < 2:
            p.error("--per-class must be at least 2")
        result = prepare(args.source, args.frozen, args.out, args.per_class)
    else:
        result = export(args.packet, args.reviews, args.frozen, args.out,
                        label_policy=args.label_policy, source_labels=args.source_labels)
    print(json.dumps(result, ensure_ascii=False, indent=2))
