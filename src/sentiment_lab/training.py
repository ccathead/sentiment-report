"""Single-GPU Transformers Trainer + PEFT LoRA; no external training framework."""
import argparse
import math
import os
import re
from pathlib import Path

from .adapters import model_spec
from .core import ROOT, code_state, environment, file_hash, load_config, now, object_hash, read_json, write_json
from .training_data import AnswerOnlyCollator, gradient_report, language_lora_targets, load_training_data


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def settings(path):
    cfg = read_json(path)
    required = {"base_config", "dataset", "evaluation_dataset", "lora_rank", "lora_alpha", "lora_dropout",
                "learning_rate", "epochs", "gradient_accumulation_steps", "warmup_ratio", "max_length", "seed"}
    if not required <= set(cfg) or set(cfg) - required - {"training_dtype"}:
        raise ValueError("Training config has missing or unknown fields")
    cfg.setdefault("training_dtype", "bfloat16")
    if cfg["training_dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Training uses bfloat16 on A800, or float32; FP16 inference is configured separately")
    for key in ("lora_rank", "lora_alpha", "gradient_accumulation_steps", "max_length", "seed"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "epochs", "lora_dropout", "warmup_ratio"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid {key}")
    if cfg["learning_rate"] <= 0 or cfg["epochs"] <= 0 or not 0 <= cfg["lora_dropout"] < 1 or not 0 <= cfg["warmup_ratio"] < 1:
        raise ValueError("Invalid optimizer or LoRA settings")
    base = load_config(project_path(cfg["base_config"]))
    if base["backend"] != "hf" or base.get("adapter_path") or not re.fullmatch(r"[a-f0-9]{40}", base.get("revision", "")):
        raise ValueError("Training requires an unadapted, pinned local HF base model")
    return cfg, base


def train(config_path, out=None, *, check_only=False, smoke=False, resume=None):
    cfg, base = settings(config_path)
    dataset = project_path(cfg["dataset"])
    data, manifest = load_training_data(dataset, project_path(cfg["evaluation_dataset"]), ROOT / "prompts/baseline.txt")
    if check_only:
        print({"status": "passed", "train": len(data["train"]), "val": len(data["val"]),
               "label_policy": manifest["label_policy"], "base_model": model_spec(base),
               "training_dtype": cfg["training_dtype"]}, flush=True)
        return
    if out is None:
        raise ValueError("--out required for training")
    out = Path(out).resolve()
    if smoke and resume:
        raise ValueError("Use a fresh output directory for smoke training")
    code = code_state()
    spec = {"training_config": cfg, "base_config": base, "smoke_only": smoke,
            "dataset_manifest_sha256": file_hash(dataset / "split_manifest.json"),
            "evaluation_manifest_sha256": file_hash(project_path(cfg["evaluation_dataset"]) / "manifest.sha256.json"),
            "source_sha256": code["source_sha256"], "environment": environment()}
    signature = object_hash(spec)
    if resume:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_relative_to(out) or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume requires a Trainer checkpoint inside the original output directory")
        info = read_json(out / "run.json")
        if info.get("state") == "completed" or (out / "adapter").exists():
            raise ValueError("Completed adapter/output must be preserved; use a new training output directory")
        if info["signature"] != signature:
            raise ValueError("Resume refused: code/config/data changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        info = {**spec, "signature": signature, "created_at": now(), "code": code, "environment": environment()}
    info.update(state="preparing", updated_at=now())
    write_json(out / "run.json", info)
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
        from .backends import HFBackend

        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or int(os.environ.get("WORLD_SIZE", "1")) != 1):
            raise ValueError("Select one CUDA GPU; this entry supports single-process training only")
        if cfg["training_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use float32 training or a BF16-capable GPU")
        set_seed(cfg["seed"])
        backend = HFBackend({**base, "dtype": cfg["training_dtype"], "seed": cfg["seed"]})
        targets = language_lora_targets(backend.model)
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"], target_modules=targets, bias="none",
                          revision=base["revision"])
        model = get_peft_model(backend.model, lora)
        model.peft_config["default"].base_model_name_or_path = base["model"]
        model.config.use_cache = False
        model.config.get_text_config().use_cache = False
        model.generation_config.use_cache = False
        model.enable_input_require_grads()
        # Only the small LoRA tensors are trainable; keep them in FP32.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        embedding_rows = model.get_input_embeddings().weight.shape[0]
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        if not trainable or any("lora_" not in name or not any(t + ".lora_" in name for t in targets) for name in trainable):
            raise ValueError("Unexpected trainable base/vision/projector parameters")
        collator = AnswerOnlyCollator(backend.processor, cfg["max_length"])
        selected = {s: rows[:2] if smoke else rows for s, rows in data.items()}
        checks = []
        # Fail on unsafe masking/oversized inputs before spending training steps.
        for split, rows in selected.items():
            for i, row in enumerate(rows, 1):
                collator([row])
                checks.append({"split": split, **collator.last_check})
                if i % 25 == 0 or i == len(rows):
                    print(f"Validated tokens/mask: {split} {i}/{len(rows)}", flush=True)
        write_json(out / "supervision-checks.json", checks)
        write_json(out / "trainable-parameters.json", {"targets": targets, "trainable": trainable,
                   "count": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        args = TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if smoke else cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"], num_train_epochs=cfg["epochs"], max_steps=2 if smoke else -1,
            warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type="cosine", optim="adamw_torch",
            bf16=cfg["training_dtype"] == "bfloat16", fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no" if smoke else "epoch", save_strategy="no" if smoke else "epoch",
            load_best_model_at_end=not smoke, metric_for_best_model="eval_loss", greater_is_better=False,
            save_total_limit=3, logging_steps=1, report_to="none", disable_tqdm=True,
            logging_nan_inf_filter=False,
            remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
            seed=cfg["seed"], data_seed=cfg["seed"], push_to_hub=False)

        class GradientCheck(TrainerCallback):
            confirmed = False
            reports = None

            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ("loss", "eval_loss", "grad_norm"):
                    if key in (logs or {}) and not math.isfinite(float(logs[key])):
                        raise ValueError(f"Nonfinite training metric: {key}")

            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                if not self.confirmed:
                    report = {"step": state.global_step, **gradient_report(model)}
                    self.reports = (self.reports or []) + [report]
                    write_json(out / "gradient-diagnostics.json", self.reports)
                    if report["status"] == "missing":
                        raise ValueError(f"No connected LoRA gradients: {report}")
                    if report["status"] == "nonfinite":
                        raise ValueError(f"NaN/Inf LoRA gradients; inspect gradient-diagnostics.json: {report}")
                    if report["status"] == "all_zero":
                        if state.global_step >= 7:
                            raise ValueError("LoRA gradients stayed zero for 8 attempts; inspect gradient-diagnostics.json")
                        print("Zero LoRA gradients on this batch; checking subsequent batches.", flush=True)
                    else:
                        self.confirmed = True
                        print("Backward check: finite nonzero LoRA gradients; base parameters frozen.", flush=True)

        gradient_check = GradientCheck()
        trainer = Trainer(model=model, args=args, train_dataset=selected["train"],
                          eval_dataset=None if smoke else selected["val"], data_collator=collator,
                          processing_class=backend.processor, callbacks=[gradient_check])
        info.update(state="training", backend=backend.info)
        write_json(out / "run.json", info)
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if not gradient_check.confirmed:
            raise ValueError("No finite nonzero LoRA gradient was observed; no successful adapter will be saved")
        if not smoke and (trainer.state.best_metric is None or not math.isfinite(trainer.state.best_metric)):
            raise ValueError("No finite validation loss checkpoint selected")
        # Trainer has restored the best validation-loss checkpoint for formal runs.
        adapter = out / "adapter"
        if model.get_input_embeddings().weight.shape[0] != embedding_rows:
            raise ValueError("Vocabulary changed unexpectedly; adapter-only export is unsafe")
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        metadata = {"base_model": model_spec(base), "processor_sha256": backend.processor_sha256,
                    "smoke_only": smoke, "training_dtype": cfg["training_dtype"],
                    "training_signature": signature, "label_policy": manifest["label_policy"],
                    "dataset_manifest_sha256": spec["dataset_manifest_sha256"], "best_checkpoint": trainer.state.best_model_checkpoint,
                    "best_validation_loss": trainer.state.best_metric, "completed_at": now()}
        write_json(adapter / "training_metadata.json", metadata)
        if not smoke:
            try:
                adapter_path = adapter.relative_to(ROOT).as_posix()
            except ValueError:
                adapter_path = str(adapter)
            write_json(out / "evaluation-config.json", {**base, "adapter_path": adapter_path})
        info.update(state="completed", completed_at=now(), training_metrics=result.metrics, adapter=str(adapter),
                    best_checkpoint=trainer.state.best_model_checkpoint, best_validation_loss=trainer.state.best_metric)
        write_json(out / "run.json", info)
        print(f"Training completed. Adapter: {adapter}", flush=True)
    except BaseException as exc:
        info.update(state="failed", updated_at=now(), error_type=type(exc).__name__, error=str(exc))
        write_json(out / "run.json", info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate data and isolation; no GPU or weights needed")
    parser.add_argument("--smoke", action="store_true", help="Two optimizer steps; never a formal model")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    train(args.config, args.out, check_only=args.check_only, smoke=args.smoke, resume=args.resume_from_checkpoint)
