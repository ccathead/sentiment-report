from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .backends import BackendError, make_backend
from .core import (code_state, environment, file_hash, index_rows, load_config, load_inputs,
                   now, object_hash, parse_label, read_json, read_jsonl, render_prompt, within, write_json)


def run_experiment(*, dataset, config_path, prompt_path, split, condition, out,
                   limit=None, dry_run=False, resume=False, protocol=None, backend_factory=make_backend):
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    cfg = load_config(config_path, allow_placeholder=dry_run)
    inputs = load_inputs(dataset, split)
    if limit is not None:
        if split != "dev" or limit < 1 or limit > len(inputs):
            raise ValueError("--limit is allowed only for 1..N development samples")
        inputs = inputs[:limit]
    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt_sha = file_hash(prompt_path)
    manifest_sha = file_hash(dataset / "manifest.sha256.json")
    if split != "dev" and not dry_run:
        if protocol is None:
            raise ValueError("Freeze the experiment protocol after dev checks, then pass --protocol for test/challenge")
        frozen = read_json(protocol)
        if frozen["dataset_manifest_sha256"] != manifest_sha:
            raise ValueError("Protocol dataset mismatch")
        if object_hash(cfg) not in frozen["config_hashes"] or prompt_sha not in frozen["prompt_hashes"]:
            raise ValueError("Configuration or prompt changed after protocol freeze")
        if condition not in frozen["conditions"]:
            raise ValueError("Condition absent from frozen protocol")
    code = code_state()
    spec = {"config": cfg, "prompt_sha256": prompt_sha, "dataset_manifest_sha256": manifest_sha,
            "split": split, "condition": condition, "selected_ids": [r["id"] for r in inputs],
            "source_sha256": code["source_sha256"], "dry_run": dry_run,
            "protocol_sha256": file_hash(protocol) if protocol else None}
    signature = object_hash(spec)
    if resume:
        if dry_run:
            raise ValueError("Dry-run previews cannot be resumed")
        info = read_json(out / "run.json")
        if info["signature"] != signature:
            raise ValueError("Resume refused: code/config/prompt/data/selection/protocol changed")
        prior = read_jsonl(out / "predictions.jsonl") if (out / "predictions.jsonl").exists() else []
        done = index_rows(prior)
        if set(done) - set(spec["selected_ids"]):
            raise ValueError("Unknown prediction ids in resume file")
        info.setdefault("resumed_at", []).append(now())
    else:
        out.mkdir(parents=True, exist_ok=False)
        info = {**spec, "signature": signature, "created_at": now(), "prompt_name": Path(prompt_path).stem,
                "code": code, "environment": environment(), "command": sys.argv,
                "working_directory": str(Path.cwd()), "state": "prepared"}
        done = {}
        (out / "prompt_template.txt").write_text(template, encoding="utf-8")
        write_json(out / "config.json", cfg)
        if protocol:
            write_json(out / "protocol.json", read_json(protocol))
        with (out / "requests.jsonl").open("x", encoding="utf-8") as output:
            for row in inputs:
                record = {"id": row["id"], "prompt": render_prompt(template, row["text"], condition),
                          "image": row["image"] if condition == "multimodal" else None,
                          "image_sha256": file_hash(within(dataset, row["image"])) if condition == "multimodal" else None}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(out / "run.json", info)
    if dry_run:
        info.update(state="dry_run", finished_at=now())
        write_json(out / "run.json", info)
        return info
    pending = [r for r in inputs if r["id"] not in done]
    if not pending:
        info.update(state="completed", finished_at=now(), recorded_samples=len(done))
        write_json(out / "run.json", info)
        return info
    try:
        backend = backend_factory(cfg)
        info.update(state="running", backend_info=backend.info)
        write_json(out / "run.json", info)
        with (out / "predictions.jsonl").open("a", encoding="utf-8") as output:
            for row in pending:
                prompt = render_prompt(template, row["text"], condition)
                image_path = within(dataset, row["image"]) if condition == "multimodal" else None
                result = {"id": row["id"], "started_at": now(), "prompt_sha256": object_hash(prompt)}
                started = time.monotonic()
                fatal = False
                try:
                    result.update(backend.infer(prompt, image_path))
                    prediction = parse_label(result.get("raw_answer"))
                    if result.get("provider_status") == "incomplete" or result.get("finish_reason") in ("length", "content_filter"):
                        prediction = None
                    result.update(prediction=prediction, status="ok" if prediction else "invalid_output")
                except BackendError as exc:
                    result.update(prediction=None, status="error", error=str(exc), attempts=exc.attempts)
                    fatal = exc.fatal
                except Exception as exc:
                    # Unexpected implementation/CUDA failures stop the run instead of fabricating 47 failed calls.
                    result.update(prediction=None, status="error", error=f"{type(exc).__name__}: {exc}")
                    fatal = True
                result["seconds"] = time.monotonic() - started
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                done[row["id"]] = result
                print(f"[{len(done)}/{len(inputs)}] {row['id']}: {result['status']}", flush=True)
                if fatal:
                    raise RuntimeError(f"Run stopped after {row['id']}; inspect the recorded error before starting a new run")
        info.update(state="completed", finished_at=now(), recorded_samples=len(done))
    except BaseException as exc:
        info.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    failure_type=type(exc).__name__, recorded_samples=len(done), stopped_at=now())
        raise
    finally:
        write_json(out / "run.json", info)
    return info
