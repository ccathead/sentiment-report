"""Create a single evaluation split without changing historical data or results."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentiment_lab.core import file_hash, load_inputs, load_gold, write_json, within


def consolidate(source, out):
    if out.exists():
        raise FileExistsError(out)
    rows = load_inputs(source, "all")
    gold = load_gold(source, "all", rows)
    (out / "inputs").mkdir(parents=True)
    (out / "labels").mkdir()
    (out / "images").mkdir()
    for row in rows:
        shutil.copy2(within(source, row["image"]), out / row["image"])
    for folder, items in (("inputs", rows), ("labels", list(gold.values()))):
        (out / folder / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    write_json(out / "metadata.json", {"date": "2026-10-31", "split_counts": {"eval": len(rows)},
        "parent_manifest_sha256": file_hash(source / "manifest.sha256.json"),
        "purpose": "Fixed regression evaluation; same ordered inputs and gold as historical all run; previously inspected"})
    write_json(out / "manifest.sha256.json", {"files": {p.relative_to(out).as_posix(): file_hash(p)
               for p in sorted(out.rglob("*")) if p.is_file()}})
    assert load_inputs(out, "eval") == rows
    assert load_gold(out, "eval", rows) == gold
    return {"eval": len(rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    p.add_argument("--out", type=Path, default=ROOT / "datasets/mvsa_eval_v1")
    args = p.parse_args()
    print(json.dumps(consolidate(args.source, args.out)))
