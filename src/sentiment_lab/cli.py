from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .core import (ROOT, SPLITS, file_hash, load_config, load_env, load_gold, load_inputs,
                   now, object_hash, read_json, write_json)


def prompt_path(value):
    path = Path(value)
    return path if path.is_file() else ROOT / "prompts" / (value + ".txt")


def parser():
    p = argparse.ArgumentParser(description="Image-text sentiment experiment runner")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a local model or DMX API; --dry-run never loads/calls a model")
    run.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--prompt", default="baseline")
    run.add_argument("--split", choices=SPLITS, default="dev")
    run.add_argument("--condition", choices=("text", "multimodal"), default="multimodal")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--limit", type=int, help="Development-only smoke test size")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--protocol", type=Path)
    run.add_argument("--env-file", type=Path)
    full = commands.add_parser("run-all", help="Run baseline image-text inference on every frozen sample, then score automatically")
    full.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    full.add_argument("--config", type=Path, required=True)
    full.add_argument("--out", type=Path, required=True)
    full.add_argument("--resume", action="store_true")
    full.add_argument("--env-file", type=Path)
    evaluate = commands.add_parser("evaluate", help="Score against separate frozen human labels")
    evaluate.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--out", type=Path)
    compare = commands.add_parser("compare", help="Compare runs on identical samples; first run is paired baseline")
    compare.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--out", type=Path, required=True)
    models = commands.add_parser("models", help="List model ids visible to the DMX account; no inference")
    models.add_argument("--config", type=Path, default=ROOT / "configs/dmx-responses.example.json")
    models.add_argument("--env-file", type=Path)
    models.add_argument("--out", type=Path)
    lock = commands.add_parser("lock-model", help="Resolve a public HF revision into a fixed commit; no weight download")
    lock.add_argument("--config", type=Path, required=True)
    lock.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check-data", help="Verify all frozen file hashes and input/label alignment")
    check.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    freeze = commands.add_parser("freeze", help="Freeze selected configurations/prompts after development checks")
    freeze.add_argument("--dataset", type=Path, default=ROOT / "datasets/mvsa_daily_v1")
    freeze.add_argument("--configs", type=Path, nargs="+", required=True)
    freeze.add_argument("--prompts", nargs="+", default=["baseline", "evidence"])
    freeze.add_argument("--conditions", nargs="+", choices=("text", "multimodal"), default=["text", "multimodal"])
    freeze.add_argument("--note", required=True, help="Dev evidence, selected comparisons, and rationale")
    freeze.add_argument("--out", type=Path, required=True)
    return p


def create_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(obj, output, ensure_ascii=False, indent=2)
        output.write("\n")


def dispatch(args):
    if args.command == "run-all":
        from .runner import run_experiment
        from .evaluation import evaluate_run, save_evaluation
        load_env(args.env_file)
        # Validate labels separately before loading a costly model; inference still reads inputs only.
        inputs = load_inputs(args.dataset, "all")
        load_gold(args.dataset, "all", inputs)
        evaluation_out = args.out / "evaluation"
        if evaluation_out.exists():
            raise FileExistsError(f"Evaluation already exists at {evaluation_out}; keep it and use a new --out for a new run")
        print(f"Full-dataset baseline: {len(inputs)} samples, multimodal input, baseline prompt. Model loads once.", flush=True)
        info = run_experiment(dataset=args.dataset, config_path=args.config, prompt_path=prompt_path("baseline"),
                              split="all", condition="multimodal", out=args.out, resume=args.resume)
        metrics, rows = evaluate_run(args.dataset, args.out)
        save_evaluation(evaluation_out, metrics, rows)
        print(json.dumps({"state": info["state"], "samples": metrics["n"], "accuracy": metrics["accuracy"],
                          "macro_f1": metrics["macro_f1"], "invalid_or_missing": metrics["invalid_or_missing"],
                          "scope": metrics["scope"], "summary": str(evaluation_out / "summary.md")}, ensure_ascii=False), flush=True)
    elif args.command == "run":
        from .runner import run_experiment
        load_env(args.env_file)
        info = run_experiment(dataset=args.dataset, config_path=args.config, prompt_path=prompt_path(args.prompt),
                              split=args.split, condition=args.condition, out=args.out, limit=args.limit,
                              dry_run=args.dry_run, resume=args.resume, protocol=args.protocol)
        print(json.dumps({"state": info["state"], "samples": len(info["selected_ids"]), "out": str(args.out)}, ensure_ascii=False))
    elif args.command == "evaluate":
        from .evaluation import evaluate_run, save_evaluation
        metrics, rows = evaluate_run(args.dataset, args.run)
        save_evaluation(args.out or args.run / "evaluation", metrics, rows)
        print(json.dumps({k: metrics[k] for k in ("n", "accuracy", "macro_f1", "invalid_or_missing", "scope")}, ensure_ascii=False))
    elif args.command == "compare":
        from .evaluation import compare_runs
        if len(args.runs) < 2:
            raise ValueError("Supply at least two runs")
        compare_runs(args.dataset, args.runs, args.out)
        print(f"Comparison saved to {args.out}")
    elif args.command == "models":
        from .backends import DMXBackend
        load_env(args.env_file)
        cfg = load_config(args.config, allow_placeholder=True)
        if cfg["backend"] != "dmx":
            raise ValueError("models requires a DMX config")
        response, _ = DMXBackend(cfg).request("GET", "models")
        rows = [{k: row[k] for k in ("id", "owned_by", "supported_endpoint_types") if k in row}
                for row in response.get("data", [])]
        if not rows:
            raise ValueError("Provider returned no model list; check account token and route")
        if args.out:
            create_json(args.out, {"queried_at": now(), "models": rows})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.command == "lock-model":
        cfg = load_config(args.config)
        if cfg["backend"] != "hf":
            raise ValueError("lock-model requires an HF config")
        model = urllib.parse.quote(cfg["model"], safe="/")
        revision = urllib.parse.quote(cfg.get("revision", "main"), safe="")
        request = urllib.request.Request(f"https://huggingface.co/api/models/{model}/revision/{revision}",
                                         headers={"User-Agent": "sentiment-report/0.1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            sha = json.load(response)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Hugging Face returned an invalid revision")
        cfg["revision"] = sha
        create_json(args.out, cfg)
        print(f"Pinned {cfg['model']} at {sha}")
    elif args.command == "check-data":
        from .core import within
        manifest = read_json(args.dataset / "manifest.sha256.json")["files"]
        for relative, sha in manifest.items():
            if file_hash(within(args.dataset, relative)) != sha:
                raise ValueError(f"Frozen file changed: {relative}")
        seen = set()
        sizes = {}
        for split in SPLITS:
            inputs = load_inputs(args.dataset, split)
            gold = load_gold(args.dataset, split, inputs)
            if seen & set(gold):
                raise ValueError("Overlapping ids across splits")
            seen.update(gold)
            sizes[split] = len(inputs)
        print(json.dumps({"status": "passed", "checked_files": len(manifest), "splits": sizes,
                          "manifest_sha256": file_hash(args.dataset / "manifest.sha256.json")}, ensure_ascii=False))
    elif args.command == "freeze":
        configs = [load_config(p) for p in args.configs]
        if not args.note.strip():
            raise ValueError("Protocol note cannot be blank")
        for cfg in configs:
            if cfg["backend"] == "hf" and not re.fullmatch(r"[0-9a-f]{40}", cfg.get("revision", "")):
                raise ValueError("Lock each HF revision before freezing the protocol")
        prompts = [prompt_path(p) for p in args.prompts]
        obj = {"created_at": now(), "note": args.note, "conditions": args.conditions,
               "dataset_manifest_sha256": file_hash(args.dataset / "manifest.sha256.json"),
               "configs": configs, "config_hashes": [object_hash(c) for c in configs],
               "prompts": {p.name: p.read_text(encoding="utf-8") for p in prompts},
               "prompt_hashes": [file_hash(p) for p in prompts],
               "policy": "Prompts/configs frozen before test/challenge; all allowed comparisons must be reported, including no improvement."}
        create_json(args.out, obj)
        print(f"Protocol frozen: {args.out}")


def main():
    args = parser().parse_args()
    try:
        dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted; flushed sample records are retained for resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
