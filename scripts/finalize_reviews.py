"""Freeze a NEW dataset version from explicit human-review exports."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
VALID={"positive","neutral","negative"}


def write_jsonl(path,rows):
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=PROJECT/"datasets"/"mvsa_daily_v0_1")
    p.add_argument("--reviews",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--min-test",type=int,default=30)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Output version already exists; never overwrite a frozen dataset")
    annotations_path=a.dataset/"annotations.jsonl"
    digest=hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    original=[json.loads(x) for x in annotations_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    reviews=json.loads(a.reviews.read_text(encoding="utf-8-sig"))
    if reviews.get("annotations_sha256")!=digest:raise ValueError("Review export does not match this dataset snapshot")
    known={r["id"] for r in original};by_id={}
    for r in reviews.get("reviews",[]):
        if r["id"] not in known or r["id"] in by_id:raise ValueError("Unknown or duplicate review ID")
        if r.get("label") not in VALID|{"exclude"} or not r.get("reviewer","").strip() or not r.get("reviewed_at"):raise ValueError("Incomplete review record")
        if r["label"]=="exclude" and not r.get("notes","").strip():raise ValueError("Exclusion requires a reason")
        by_id[r["id"]]=r
    missing=[r["id"] for r in original if r["split"]!="challenge" and r["id"] not in by_id]
    if missing:raise ValueError(f"Main samples pending human review: {len(missing)}; first IDs: {missing[:6]}")
    accepted=[];excluded=[];pending=[]
    for r in original:
        s=by_id.get(r["id"])
        if s is None:pending.append(r);continue
        updated={**r,"human_final_label":s["label"] if s["label"] in VALID else None,"human_review_status":"excluded" if s["label"]=="exclude" else "reviewed","human_reviewer":s["reviewer"],"human_notes":s.get("notes",""),"human_reviewed_at":s["reviewed_at"]}
        (excluded if s["label"]=="exclude" else accepted).append(updated)
    test=[r for r in accepted if r["split"]=="test"]
    dev=[r for r in accepted if r["split"]=="dev"]
    if len(test)<a.min_test:raise ValueError(f"Only {len(test)} reviewed test samples; require at least {a.min_test}")
    if set(r["human_final_label"] for r in test)!=VALID:raise ValueError("Reviewed test set must cover all three labels")
    if set(r["human_final_label"] for r in dev)!=VALID:raise ValueError("Reviewed development set must cover all three labels")
    a.out.mkdir(parents=True)
    for name in ["images","source_text","inputs","labels"]:(a.out/name).mkdir()
    for r in accepted+pending:
        for field in ["image","raw_text_file"]:
            target=a.out/r[field]
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(a.dataset/r[field],target)
    write_jsonl(a.out/"annotations.jsonl",accepted)
    write_jsonl(a.out/"excluded.jsonl",excluded)
    write_jsonl(a.out/"challenge_pending.jsonl",pending)
    for split in ["dev","test","challenge"]:
        rows=[r for r in accepted if r["split"]==split]
        write_jsonl(a.out/"inputs"/(split+".jsonl"),[{"id":r["id"],"image":r["image"],"text":r["text"]} for r in rows])
        write_jsonl(a.out/"labels"/(split+".jsonl"),[{"id":r["id"],"label":r["human_final_label"]} for r in rows])
    shutil.copyfile(a.reviews,a.out/"human_review_export.json")
    meta={"dataset_name":"MVSA-Daily-Course","version":a.out.name,"parent_annotations_sha256":digest,"review_export_sha256":hashlib.sha256(a.reviews.read_bytes()).hexdigest(),"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"status":"human-reviewed snapshot; single-rater unless independently reviewed","split_counts":dict(Counter(r["split"] for r in accepted)),"label_counts_by_split":{s:dict(Counter(r["human_final_label"] for r in accepted if r["split"]==s)) for s in ["dev","test","challenge"]},"excluded_count":len(excluded),"pending_challenge_count":len(pending)}
    (a.out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
