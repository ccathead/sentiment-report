"""Local LoRA identity and integrity checks; no torch/PEFT import required."""
from pathlib import Path

from .core import ROOT, file_hash, read_json


def model_spec(cfg):
    result = {k: cfg[k] for k in ("model", "revision", "architecture")}
    result["dtype"] = cfg.get("dtype", "float16")
    if cfg["architecture"] == "qwen2_5_vl":
        result.update(min_pixels=cfg.get("min_pixels", 200704), max_pixels=cfg.get("max_pixels", 1003520))
    return result


def adapter_snapshot(cfg):
    if not cfg.get("adapter_path"):
        return None
    path = Path(cfg["adapter_path"])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    names = ("adapter_config.json", "adapter_model.safetensors", "training_metadata.json")
    if not all((path / name).is_file() for name in names):
        raise ValueError("Local adapter requires config, safetensors and training_metadata.json")
    meta, peft = read_json(path / names[2]), read_json(path / names[0])
    if meta.get("base_model") != model_spec(cfg):
        raise ValueError("Adapter base model/revision/image configuration mismatch")
    if peft.get("base_model_name_or_path") != cfg["model"] or peft.get("revision") != cfg["revision"]:
        raise ValueError("PEFT config base model/revision mismatch")
    if meta.get("smoke_only"):
        raise ValueError("Smoke adapters cannot be used for formal evaluation")
    return {"path": str(path), "files": {name: file_hash(path / name) for name in names}, "metadata": meta}
