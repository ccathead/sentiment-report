"""Build a traceable MVSA subset with AI suggestions and unfilled human labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
LABELS = {"positive", "negative", "neutral"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")


def normalise_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def text_group(text: str) -> str:
    text = re.sub(r"https?://\S+", "[url]", text.lower())
    text = re.sub(r"@[\w_]+", "@user", text)
    return sha(" ".join(text.split()).encode("utf-8"))


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        pixels = list(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=PROJECT/"configs"/"curation_v0_1.json")
    parser.add_argument("--out", type=Path, default=PROJECT/"datasets"/"mvsa_daily_v0_1")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.out.exists():
        raise FileExistsError(f"Never overwrite a dataset version: {args.out}")
    source_labels = {}
    label_file = args.source/"labelResultAll.txt"
    for line in label_file.read_text(encoding="utf-8-sig").splitlines()[1:]:
        key, values = line.split("\t")
        source_labels[key] = values.split(",")
    by_label = {label: [] for label in sorted(LABELS)}
    for row in spec["main"]:
        by_label[row[1]].append(str(row[0]))
    rng = random.Random(spec["seed"])
    dev_ids = set()
    for label, ids in by_label.items():
        ids.sort(key=int)
        rng.shuffle(ids)
        dev_ids.update(ids[:spec["development_per_proposed_class"]])
    all_rows = [("main", row) for row in spec["main"]]+[("challenge", row) for row in spec["challenge"]]
    ids = [str(row[0]) for _, row in all_rows]
    assert len(ids) == len(set(ids)), "Duplicate source IDs"
    args.out.mkdir(parents=True)
    for name in ["images", "source_text", "inputs", "labels"]:
        (args.out/name).mkdir()
    annotations = []
    image_hashes = {}
    text_hashes = {}
    perceptual = {}
    for role, raw_row in all_rows:
        item = dict(zip(spec["record_fields"], raw_row))
        key = str(item["source_id"])
        image_source = args.source/"data"/f"{key}.jpg"
        text_source = args.source/"data"/f"{key}.txt"
        image_bytes, text_bytes = image_source.read_bytes(), text_source.read_bytes()
        text = normalise_text(text_bytes.decode("utf-8-sig"))
        assert text.strip()
        with Image.open(image_source) as im:
            im.load()
            width, height = im.size
        image_rel, text_rel = f"images/mvsa_{key}.jpg", f"source_text/mvsa_{key}.txt"
        shutil.copyfile(image_source, args.out/image_rel)
        shutil.copyfile(text_source, args.out/text_rel)
        split = "challenge" if role == "challenge" else ("dev" if key in dev_ids else "test")
        original = source_labels[key]
        annotation = {
            "id": f"mvsa_{key}", "source_id": key, "source_dataset": "MVSA-Single",
            "source_mirror": "Kaggle vincemarcs/mvsasingle version 1",
            "split": split, "image": image_rel, "text": text,
            "raw_text_file": text_rel, "language": "en", "domain": item["domain"],
            "image_sha256": sha(image_bytes), "raw_text_sha256": sha(text_bytes),
            "normalised_text_sha256": sha(text.encode("utf-8")),
            "image_width": width, "image_height": height,
            "source_text_label": original[0], "source_image_label": original[1],
            "source_consensus_label": original[0] if original[0] == original[1] else None,
            "ai_proposed_joint_label": item["proposed_label"],
            "ai_confidence": item["confidence"],
            "visible_evidence_zh": item["visible_evidence_zh"],
            "annotation_rationale_zh": item["annotation_rationale_zh"],
            "annotation_provenance": "Codex-assisted image-text review; source labels available to curator",
            "human_final_label": None, "human_review_status": "pending",
            "human_reviewer": "", "human_notes": "", "human_reviewed_at": None,
            "source_label_disagreement": original[0] != original[1],
            "proposed_differs_from_source_text": item["proposed_label"] is not None and item["proposed_label"] != original[0],
            "unresolved_joint_polarity": item["proposed_label"] is None,
        }
        annotations.append(annotation)
        image_hashes.setdefault(annotation["image_sha256"], []).append(annotation["id"])
        text_hashes.setdefault(text_group(text), []).append(annotation["id"])
        perceptual[annotation["id"]] = dhash(image_source)
    duplicates = {"exact_images": [x for x in image_hashes.values() if len(x)>1],
                  "normalised_text": [x for x in text_hashes.values() if len(x)>1]}
    assert not duplicates["exact_images"] and not duplicates["normalised_text"], duplicates
    near = []
    for i, a in enumerate(annotations):
        for b in annotations[i+1:]:
            distance = (perceptual[a["id"]] ^ perceptual[b["id"]]).bit_count()
            if distance <= 4:
                near.append({"id_a": a["id"], "id_b": b["id"], "dhash_distance": distance,
                             "cross_split": a["split"] != b["split"]})
    for split in ["dev", "test", "challenge"]:
        rows = [a for a in annotations if a["split"] == split]
        jsonl(args.out/"inputs"/f"{split}.jsonl", [{"id":a["id"],"image":a["image"],"text":a["text"]} for a in rows])
        jsonl(args.out/"labels"/f"{split}.pending.jsonl", [{"id":a["id"],"label":None,"status":"pending_human_review"} for a in rows])
    jsonl(args.out/"annotations.jsonl", annotations)
    with (args.out/"annotations_review.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    spec_copy = args.out/"curation_spec.json"
    shutil.copyfile(args.spec,spec_copy)
    metadata = {
        "dataset_name":"MVSA-Daily-Course", "version":spec["version"],
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"provisional; human joint-label review required before accuracy evaluation",
        "task":"image-text joint sentiment polarity classification",
        "label_set":sorted(LABELS), "seed":spec["seed"],
        "split_counts":dict(Counter(a["split"] for a in annotations)),
        "proposed_label_counts_by_split":{s:dict(Counter(a["ai_proposed_joint_label"] or "unresolved" for a in annotations if a["split"]==s)) for s in ["dev","test","challenge"]},
        "source_polarity_agreement_counts":dict(Counter(str(a["source_text_label"]==a["source_image_label"]) for a in annotations)),
        "human_reviewed_count":0, "human_review_pending_count":len(annotations),
        "text_normalisation":"strict UTF-8-sig decode; HTML entity unescape; collapse whitespace; preserve words, hashtags, URLs, handles; no translation",
        "image_processing":"byte-for-byte copy, no crop or resize",
        "source_archive_sha256":"743963706a9d5722180ccfd156a165245a526d5f18a58a419ae867d5c5b69fc4",
        "source_label_file_sha256":sha(label_file.read_bytes()),
        "curation_spec_sha256":sha(args.spec.read_bytes()),
        "annotations_sha256":sha((args.out/"annotations.jsonl").read_bytes()),
        "duplicates":duplicates,"near_duplicate_candidates":near,
        "test_blinding":"No evaluated-model predictions were used for selection. The curator viewed test items for annotation; this is not an unseen external benchmark.",
        "source_authorship":"Public MVSA material; this project screens, organises and proposes new joint labels, not original collection of the Twitter posts.",
    }
    (args.out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
