"""Validate the local provisional dataset without treating AI labels as gold."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=PROJECT/"datasets"/"mvsa_daily_v0_1")
    a=p.parse_args();root=a.dataset.resolve()
    annotations=load(root/"annotations.jsonl")
    metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    assert len(annotations)==72 and len({r["id"] for r in annotations})==72
    assert hashlib.sha256((root/"annotations.jsonl").read_bytes()).hexdigest()==metadata["annotations_sha256"]
    assert Counter(r["split"] for r in annotations)=={"dev":12,"test":48,"challenge":12}
    for r in annotations:
        assert r["human_final_label"] is None and r["human_review_status"]=="pending"
        for field,expected in [("image",r["image_sha256"]),("raw_text_file",r["raw_text_sha256"])]:
            target=(root/r[field]).resolve()
            assert target.is_relative_to(root)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
        with Image.open(root/r["image"]) as image:image.load()
        assert hashlib.sha256(r["text"].encode("utf-8")).hexdigest()==r["normalised_text_sha256"]
    for split in ["dev","test","challenge"]:
        inputs=load(root/"inputs"/(split+".jsonl"))
        pending=load(root/"labels"/(split+".pending.jsonl"))
        expected={r["id"]:r for r in annotations if r["split"]==split}
        assert {r["id"] for r in inputs}==set(expected)=={r["id"] for r in pending}
        for r in inputs:
            assert set(r)=={"id","image","text"},"Answer or annotation metadata leaked into inference input"
            assert r["image"]==expected[r["id"]]["image"] and r["text"]==expected[r["id"]]["text"]
        assert all(r["label"] is None for r in pending)
    for name in ["review.html","README.md","ANNOTATION_GUIDE.md","annotations_review.csv"]:assert (root/name).is_file()
    result={"status":"passed","samples":72,"split_counts":metadata["split_counts"],"image_and_text_hashes":"passed","inference_inputs_contain_no_labels":True,"human_gold_labels":0,"ready_for_formal_accuracy":False}
    (root/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
