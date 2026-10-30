from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("positive", "negative", "neutral")
SPLITS = ("dev", "test", "challenge")
ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def object_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{n}; preserve/repair the partial file before resuming") from exc
    return rows


def index_rows(rows):
    indexed = {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every row requires a nonempty string id")
        if row["id"] in indexed:
            raise ValueError(f"Duplicate id: {row['id']}")
        indexed[row["id"]] = row
    return indexed


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path must remain inside dataset root")
    return path


def load_inputs(root, split):
    """Read only model inputs and integrity hashes; never read annotations or gold labels."""
    root = Path(root).resolve()
    if split == "all":
        rows = [row for part in SPLITS for row in load_inputs(root, part)]
        index_rows(rows)  # Refuse overlap rather than silently counting any sample twice.
        return rows
    if split not in SPLITS:
        raise ValueError("Unknown split")
    manifest = read_json(root / "manifest.sha256.json")["files"]
    relative = f"inputs/{split}.jsonl"
    if file_hash(root / relative) != manifest.get(relative):
        raise ValueError(f"Frozen input hash mismatch: {relative}")
    rows = read_jsonl(root / relative)
    index_rows(rows)
    if not rows:
        raise ValueError("Empty input split")
    for row in rows:
        if set(row) != {"id", "image", "text"}:
            raise ValueError("Model inputs must contain ONLY id, image, text")
        if not isinstance(row["text"], str) or not isinstance(row["image"], str):
            raise ValueError("Image and text must be strings")
        path = within(root, row["image"])
        if file_hash(path) != manifest.get(row["image"]):
            raise ValueError(f"Frozen image hash mismatch: {row['id']}")
    return rows


def load_gold(root, split, inputs):
    root = Path(root)
    if split == "all":
        rows = [row for part in SPLITS
                for row in load_gold(root, part, load_inputs(root, part)).values()]
        gold = index_rows(rows)
        if set(gold) != set(index_rows(inputs)):
            raise ValueError("Input and gold ids differ")
        return gold
    relative = f"labels/{split}.jsonl"
    manifest = read_json(root / "manifest.sha256.json")["files"]
    if file_hash(root / relative) != manifest.get(relative):
        raise ValueError("Frozen gold label hash mismatch")
    gold = index_rows(read_jsonl(root / relative))
    if set(gold) != set(index_rows(inputs)):
        raise ValueError("Input and gold ids differ")
    if any(set(r) != {"id", "label"} or r["label"] not in LABELS for r in gold.values()):
        raise ValueError("Invalid gold schema or label")
    return gold


def render_prompt(template, text, condition):
    if condition not in ("text", "multimodal"):
        raise ValueError("Unknown input condition")
    return template.format(
        modality_instruction=("Use the attached image together with the post text." if condition == "multimodal"
                              else "Use only the post text. No image is available; do not assume its contents."),
        evidence_instruction=("Check whether the image supports, adds to, or conflicts with the textual attitude."
                              if condition == "multimodal" else "Check whether the textual cues support or conflict with each other."),
        post_text_json=json.dumps(text, ensure_ascii=False),
    )


def parse_label(text):
    """Exact class or single-field JSON; never search explanations for a class word."""
    if not isinstance(text, str):
        return None
    value = text.strip().lower()
    if value in LABELS:
        return value
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and set(obj) == {"label"} and isinstance(obj["label"], str):
        value = obj["label"].strip().lower()
        return value if value in LABELS else None
    return None


def load_config(path, allow_placeholder=False):
    cfg = read_json(path)
    common = {"backend", "model"}
    hf = {"architecture", "revision", "dtype", "device", "min_pixels", "max_pixels", "max_new_tokens", "seed"}
    api = {"base_url", "endpoint", "api_key_env", "max_output_tokens", "temperature", "extra_body",
           "timeout_seconds", "max_retries", "request_interval_seconds", "chat_token_parameter"}
    backend = cfg.get("backend")
    if backend not in ("hf", "dmx"):
        raise ValueError("backend must be hf or dmx")
    unknown = set(cfg) - common - (hf if backend == "hf" else api)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}; secrets must use environment variables")
    if not isinstance(cfg.get("model"), str) or not cfg["model"]:
        raise ValueError("A model id is required")
    if "REPLACE_" in cfg["model"] and not allow_placeholder:
        raise ValueError("Choose an account-verified visual model id in a local config first")
    if backend == "hf":
        if cfg.get("architecture") not in ("llava", "qwen2_5_vl"):
            raise ValueError("Unsupported HF architecture")
        if cfg.get("dtype", "float16") not in ("float16", "bfloat16"):
            raise ValueError("Supported GPU dtypes: float16, bfloat16")
        if cfg.get("max_new_tokens", 32) < 1:
            raise ValueError("max_new_tokens must be positive")
    else:
        if cfg.get("base_url") != "https://www.dmxapi.cn/v1":
            raise ValueError("This adapter sends DMX_API_KEY only to https://www.dmxapi.cn/v1")
        if cfg.get("endpoint") not in ("responses", "chat/completions"):
            raise ValueError("Unsupported DMX endpoint")
        if cfg.get("max_output_tokens", 512) < 16 or cfg.get("timeout_seconds", 120) <= 0:
            raise ValueError("Invalid output limit or timeout")
        if not 0 <= cfg.get("max_retries", 2) <= 5 or cfg.get("request_interval_seconds", 1) < 0:
            raise ValueError("Invalid retry/interval setting")
        if cfg.get("chat_token_parameter", "max_completion_tokens") not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Invalid chat token parameter")
        extra = cfg.get("extra_body", {})
        if not isinstance(extra, dict) or set(extra) - {"reasoning", "reasoning_effort", "top_p"}:
            raise ValueError("extra_body allows only reasoning, reasoning_effort, top_p; no input overrides")
    return cfg


def load_env(path):
    """Small literal KEY=value reader; does not execute shell text or overwrite existing env."""
    if not path:
        return
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Invalid .env line; expected KEY=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def code_state():
    def git(*args):
        try:
            p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / "experiment.py"]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}}


def environment():
    versions = {}
    for package in ("Pillow", "torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "numpy", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
