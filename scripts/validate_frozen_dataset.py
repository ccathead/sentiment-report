"""Check that frozen labels exactly match the user's recorded decisions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
VALID = {"positive", "negative", "neutral"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT/"datasets"/"mvsa_daily_v1")
    args = parser.parse_args()
    root = args.dataset.resolve()
    meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    annotations = read_jsonl(root/"annotations.jsonl")
    excluded = read_jsonl(root/"excluded.jsonl")
    pending = read_jsonl(root/"challenge_pending.jsonl")
    reviews_path = root/"human_review_export.json"
    reviews = json.loads(reviews_path.read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(reviews_path.read_bytes()).hexdigest() == meta["review_export_sha256"]
    by_id = {r["id"]: r for r in reviews["reviews"]}
    assert len(by_id) == len(reviews["reviews"])
    accepted_ids = {r["id"] for r in annotations}
    excluded_ids = {r["id"] for r in excluded}
    pending_ids = {r["id"] for r in pending}
    assert len(accepted_ids) == len(annotations)
    assert not (accepted_ids & excluded_ids or accepted_ids & pending_ids or excluded_ids & pending_ids)
    assert accepted_ids | excluded_ids == set(by_id)
    assert not pending_ids & set(by_id)
    image_hashes = []; text_hashes = []
    for r in annotations:
        review = by_id[r["id"]]
        assert r["human_review_status"] == "reviewed"
        assert r["human_final_label"] in VALID
        assert r["human_final_label"] == review["label"]
        assert r["human_reviewer"] == review["reviewer"] and r["human_reviewed_at"] == review["reviewed_at"]
        assert r["human_notes"] == review.get("notes", "")
        for key, expected in [("image", r["image_sha256"]), ("raw_text_file", r["raw_text_sha256"])]:
            path = (root/r[key]).resolve()
            assert path.is_relative_to(root)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        with Image.open(root/r["image"]) as image:
            image.load()
            assert image.size == (r["image_width"], r["image_height"])
        assert hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == r["normalised_text_sha256"]
        image_hashes.append(r["image_sha256"]); text_hashes.append(r["normalised_text_sha256"])
    assert len(set(image_hashes)) == len(image_hashes)
    assert len(set(text_hashes)) == len(text_hashes)
    for r in excluded:
        review = by_id[r["id"]]
        assert review["label"] == "exclude" and review.get("notes", "").strip()
        assert r["human_final_label"] is None and r["human_notes"] == review["notes"]
    for split in ["dev", "test", "challenge"]:
        expected = {r["id"]: r for r in annotations if r["split"] == split}
        inputs = read_jsonl(root/"inputs"/(split+".jsonl"))
        labels = read_jsonl(root/"labels"/(split+".jsonl"))
        assert len(inputs) == len(labels) == len(expected)
        assert {r["id"] for r in inputs} == {r["id"] for r in labels} == set(expected)
        for r in inputs:
            assert set(r) == {"id", "image", "text"}
            assert r["image"] == expected[r["id"]]["image"] and r["text"] == expected[r["id"]]["text"]
        for r in labels:
            assert set(r) == {"id", "label"}
            assert r["label"] == expected[r["id"]]["human_final_label"]
        assert len(expected) == meta["split_counts"].get(split, 0)
        assert dict(Counter(r["human_final_label"] for r in expected.values())) == meta["label_counts_by_split"][split]
    assert meta["split_counts"]["test"] >= 30
    assert all(set(meta["label_counts_by_split"][s]) == VALID for s in ["dev", "test"])
    result = {
        "status": "passed", "accepted_samples": len(annotations), "excluded_samples": len(excluded),
        "pending_samples": len(pending), "split_counts": meta["split_counts"],
        "labels_match_human_export": True, "images_and_raw_text_hashes_match": True,
        "unique_image_and_text_hashes": True, "input_label_separation": True,
        "label_sets_cover_three_classes": all(set(meta["label_counts_by_split"][s]) == VALID for s in ["dev", "test"]),
        "ready_for_inference_and_scoring": True,
    }
    (root/"validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
