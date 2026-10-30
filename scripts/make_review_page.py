"""Embed local metadata in an offline review page; no external requests."""
import argparse
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=PROJECT/"datasets"/"mvsa_daily_v0_1")
    a=p.parse_args()
    metadata=json.loads((a.dataset/"metadata.json").read_text(encoding="utf-8"))
    annotations=[json.loads(x) for x in (a.dataset/"annotations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    payload={"dataset_name":metadata["dataset_name"],"dataset_version":metadata["version"],"annotations_sha256":metadata["annotations_sha256"],"annotations":annotations}
    encoded=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c").replace("&","\\u0026")
    template=(PROJECT/"templates"/"review_template.html").read_text(encoding="utf-8")
    target=a.dataset/"review.html"
    if target.exists(): raise FileExistsError(target)
    target.write_text(template.replace("__DATA__",encoded),encoding="utf-8")
    print(target)
