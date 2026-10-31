"""Embed local metadata in an offline review page; no external requests."""
import argparse
import json
import hashlib
import re
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=PROJECT/"datasets"/"mvsa_daily_v0_1")
    p.add_argument("--refresh",action="store_true",help="Refresh layout while preserving the annotation version and browser draft key")
    a=p.parse_args()
    metadata=json.loads((a.dataset/"metadata.json").read_text(encoding="utf-8"))
    annotations=[json.loads(x) for x in (a.dataset/"annotations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    payload={"dataset_name":metadata["dataset_name"],"dataset_version":metadata["version"],"annotations_sha256":metadata["annotations_sha256"],"annotations":annotations}
    encoded=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c").replace("&","\\u0026")
    template=(PROJECT/"templates"/"review_template.html").read_text(encoding="utf-8")
    target=a.dataset/"review.html"
    actual=hashlib.sha256((a.dataset/"annotations.jsonl").read_bytes()).hexdigest()
    if metadata["annotations_sha256"]!=actual: raise ValueError("Annotation integrity mismatch")
    if target.exists():
        if not a.refresh: raise FileExistsError(target)
        old=re.search(r'<script id="dataset" type="application/json">(.*?)</script>',target.read_text(encoding="utf-8"),re.S)
        if old is None or json.loads(old.group(1))["annotations_sha256"]!=actual:
            raise ValueError("Refresh refused: annotation version changed; preserve the original review page")
    target.write_text(template.replace("__DATA__",encoded),encoding="utf-8")
    print(target)
