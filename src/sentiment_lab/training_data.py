"""Validated local SFT data and answer-only, single-image collation."""
import html
import unicodedata
from pathlib import Path

from .backends import user_messages
from .core import LABELS, file_hash, index_rows, load_inputs, read_json, render_prompt, within


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def load_training_data(dataset, evaluation, template_path):
    dataset, evaluation = Path(dataset), Path(evaluation)
    manifest = read_json(dataset / "split_manifest.json")
    if file_hash(template_path) != manifest["prompt_sha256"]:
        raise ValueError("Training task prompt changed since dataset export")
    for relative, expected in manifest["files"].items():
        if file_hash(within(dataset, relative)) != expected:
            raise ValueError(f"Training data hash mismatch: {relative}")
    if not {"train.json", "val.json", "dataset_info.json"} <= set(manifest["files"]):
        raise ValueError("Training manifest must cover both splits")
    evaluation_rows = load_inputs(evaluation, "all")  # Never read evaluation labels.
    blocked_ids = {r["id"] for r in evaluation_rows}
    blocked_images = {file_hash(within(evaluation, r["image"])) for r in evaluation_rows}
    blocked_texts = {text_key(r["text"]) for r in evaluation_rows}
    inventory = index_rows(manifest["samples"])
    if any(r["split"] not in ("train", "val") for r in inventory.values()):
        raise ValueError("Unexpected training split")
    template = Path(template_path).read_text(encoding="utf-8")
    result, seen_images, seen_texts = {}, set(), set()
    for split in ("train", "val"):
        samples = sorted((r for r in inventory.values() if r["split"] == split), key=lambda r: r["id"])
        records = read_json(dataset / f"{split}.json")
        if not samples or len(samples) != len(records):
            raise ValueError("Empty or mismatched training split")
        result[split] = []
        for sample, record in zip(samples, records):
            prompt = render_prompt(template, sample["text"], "multimodal")
            label = sample.get("training_label", sample.get("review", {}).get("label"))
            expected = {"instruction": "<image>\n" + prompt, "input": "", "output": label, "images": [sample["image"]]}
            if record != expected or label not in LABELS:
                raise ValueError("SFT content differs from its label/provenance manifest")
            image = within(dataset, sample["image"])
            image_hash, text = file_hash(image), text_key(sample["text"])
            if manifest["files"].get(sample["image"]) != image_hash or sample["image_sha256"] != image_hash:
                raise ValueError("Training image integrity mismatch")
            if (sample["id"] in blocked_ids or image_hash in blocked_images | seen_images
                    or text in blocked_texts | seen_texts):
                raise ValueError("Training/validation/evaluation overlap or duplicate")
            seen_images.add(image_hash)
            seen_texts.add(text)
            result[split].append({"id": sample["id"], "prompt": prompt, "label": label, "image": str(image)})
        if {r["label"] for r in result[split]} != set(LABELS):
            raise ValueError("Each split must contain all three classes")
    return result, manifest


def language_lora_targets(model):
    projections = {"q_proj", "k_proj", "v_proj", "o_proj"}
    forbidden = {"visual", "vision_tower", "vision_model", "multi_modal_projector"}
    names = [name for name, _ in model.named_modules()
             if name.split(".")[-1] in projections and "self_attn" in name.split(".")
             and not (set(name.split(".")) & forbidden)]
    if {n.split(".")[-1] for n in names} != projections:
        raise ValueError("Could not identify all four language attention projections")
    return sorted(names)


def gradient_report(model):
    """Inspect actual gradients without conflating disconnected, nonfinite and zero."""
    import torch

    result = {"trainable_tensors": 0, "missing_tensors": 0, "nonfinite_tensors": 0,
              "nonzero_tensors": 0, "nonfinite_examples": [], "finite_max_abs": 0.0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        result["trainable_tensors"] += 1
        grad = parameter.grad
        if grad is None:
            result["missing_tensors"] += 1
            continue
        if not bool(torch.isfinite(grad).all()):
            result["nonfinite_tensors"] += 1
            if len(result["nonfinite_examples"]) < 5:
                result["nonfinite_examples"].append(name)
            continue
        if bool(torch.count_nonzero(grad)):
            result["nonzero_tensors"] += 1
            result["finite_max_abs"] = max(result["finite_max_abs"], float(grad.detach().abs().max()))
    if result["trainable_tensors"] == result["missing_tensors"]:
        result["status"] = "missing"
    elif result["nonfinite_tensors"]:
        result["status"] = "nonfinite"
    elif result["nonzero_tensors"]:
        result["status"] = "finite_nonzero"
    else:
        result["status"] = "all_zero"
    return result


class AnswerOnlyCollator:
    """One example per microbatch avoids model-specific visual batch packing."""
    def __init__(self, processor, max_length):
        self.processor, self.max_length = processor, max_length
        self.last_check = None

    def __call__(self, examples):
        import torch
        from PIL import Image, ImageOps

        if len(examples) != 1:
            raise ValueError("This multimodal collator requires microbatch size 1")
        example = examples[0]
        messages = user_messages(example["prompt"])
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": [{"type": "text", "text": example["label"]}]}],
            tokenize=False, add_generation_prompt=False)
        with Image.open(example["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        # Tokenize both with the same image expansion and special-token policy as inference.
        prompt_inputs = self.processor(text=[prefix], images=[image], padding=True,
                                       truncation=False, return_tensors="pt")
        inputs = self.processor(text=[full], images=[image], padding=True,
                                truncation=False, return_tensors="pt")
        n = prompt_inputs["input_ids"].shape[1]
        ids = inputs["input_ids"]
        if ids.shape[1] > self.max_length:
            raise ValueError(f"{example['id']}: {ids.shape[1]} tokens exceed max_length; no truncation performed")
        if n >= ids.shape[1] or not torch.equal(ids[:, :n], prompt_inputs["input_ids"]):
            raise ValueError("Chat template/tokenizer prefix mismatch; answer masking is unsafe")
        target = self.processor.tokenizer.decode(ids[0, n:].tolist(), skip_special_tokens=True).strip()
        if target != example["label"]:
            raise ValueError("Assistant suffix does not decode to exactly the target label")
        labels = ids.clone()
        labels[:, :n] = -100
        labels[inputs["attention_mask"] == 0] = -100
        if not torch.any(labels != -100):
            raise ValueError("No supervised answer tokens")
        inputs["labels"] = labels
        self.last_check = {"id": example["id"], "input_tokens": ids.shape[1], "prompt_tokens": n,
                           "supervised_tokens": int((labels != -100).sum()), "target": target}
        return inputs
