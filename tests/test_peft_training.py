"""Synthetic data/adapter/mask tests; never download weights or call APIs."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentiment_lab.adapters import adapter_snapshot, model_spec
from sentiment_lab.core import ROOT, file_hash, load_config, object_hash, read_json, render_prompt, write_json
from sentiment_lab.training_data import AnswerOnlyCollator, gradient_report, language_lora_targets, load_training_data


class TrainingTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset, self.evaluation = self.root / "sft", self.root / "eval"
        for p in (self.dataset / "images", self.evaluation / "images", self.evaluation / "inputs"):
            p.mkdir(parents=True)
        samples = []
        template = (ROOT / "prompts/baseline.txt").read_text(encoding="utf-8")
        for split, offset in (("train", 1), ("val", 4)):
            records = []
            for i, label in enumerate(("positive", "negative", "neutral"), offset):
                relative = f"images/{i}.png"
                Image.new("RGB", (4, 4), (i * 20, 0, 0)).save(self.dataset / relative)
                sample = {"id": str(i), "image": relative, "text": f"post {i}", "split": split,
                          "training_label": label, "image_sha256": file_hash(self.dataset / relative)}
                samples.append(sample)
                records.append({"instruction": "<image>\n" + render_prompt(template, sample["text"], "multimodal"),
                                "input": "", "output": label, "images": [relative]})
            write_json(self.dataset / f"{split}.json", records)
        write_json(self.dataset / "dataset_info.json", {})
        self.manifest = {"samples": samples, "prompt_sha256": file_hash(ROOT / "prompts/baseline.txt"),
                         "files": {p.relative_to(self.dataset).as_posix(): file_hash(p)
                                   for p in self.dataset.rglob("*") if p.is_file()}}
        write_json(self.dataset / "split_manifest.json", self.manifest)
        Image.new("RGB", (4, 4), (255, 255, 255)).save(self.evaluation / "images/eval.png")
        self.evalrow = {"id": "evaluation", "image": "images/eval.png", "text": "evaluation text"}
        self.save_evaluation()
        self.base = read_json(ROOT / "configs/qwen.locked.json")

    def save_evaluation(self):
        (self.evaluation / "inputs/eval.jsonl").write_text(json.dumps(self.evalrow) + "\n", encoding="utf-8")
        write_json(self.evaluation / "manifest.sha256.json", {"files": {
            p.relative_to(self.evaluation).as_posix(): file_hash(p)
            for p in self.evaluation.rglob("*") if p.is_file() and p.name != "manifest.sha256.json"}})

    def test_dataset_load_and_no_evaluation_gold_read(self):
        data, _ = load_training_data(self.dataset, self.evaluation, ROOT / "prompts/baseline.txt")
        self.assertEqual([len(data[s]) for s in ("train", "val")], [3, 3])
        self.assertTrue(all("<image>" not in r["prompt"] for r in data["train"]))
        self.assertFalse((self.evaluation / "labels").exists())

    def test_reject_evaluation_overlap(self):
        self.evalrow["text"] = "POST 1"
        self.save_evaluation()
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_training_data(self.dataset, self.evaluation, ROOT / "prompts/baseline.txt")

    def test_reject_data_tampering(self):
        (self.dataset / "val.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_training_data(self.dataset, self.evaluation, ROOT / "prompts/baseline.txt")

    def test_targets_exclude_visual_and_projector(self):
        names = [f"model.language_model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names += ["model.vision_tower.encoder.self_attn.q_proj", "model.visual.self_attn.v_proj", "multi_modal_projector.self_attn.o_proj"]
        class Fake:
            def named_modules(self):
                return [(n, None) for n in names]
        targets = language_lora_targets(Fake())
        self.assertEqual(len(targets), 4)
        self.assertTrue(all("language_model" in n for n in targets))

    def make_adapter(self):
        path = self.root / "adapter"
        path.mkdir()
        write_json(path / "adapter_config.json", {"base_model_name_or_path": self.base["model"], "revision": self.base["revision"]})
        write_json(path / "training_metadata.json", {"base_model": model_spec(self.base), "smoke_only": False})
        (path / "adapter_model.safetensors").write_bytes(b"synthetic-placeholder-not-a-model")
        return path

    def test_adapter_snapshot_tracks_weight_mutation(self):
        path = self.make_adapter()
        config = {**self.base, "adapter_path": str(path)}
        first = adapter_snapshot(config)
        (path / "adapter_model.safetensors").write_bytes(b"changed")
        self.assertNotEqual(object_hash(first), object_hash(adapter_snapshot(config)))
        config["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "mismatch"):
            adapter_snapshot(config)

    def test_smoke_adapter_refused_for_evaluation(self):
        path = self.make_adapter()
        metadata = read_json(path / "training_metadata.json")
        metadata["smoke_only"] = True
        write_json(path / "training_metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "Smoke"):
            adapter_snapshot({**self.base, "adapter_path": str(path)})

    def test_empty_adapter_config_refused(self):
        path = self.root / "config.json"
        write_json(path, {**self.base, "adapter_path": ""})
        with self.assertRaisesRegex(ValueError, "adapter_path"):
            load_config(path)

    def test_llava_training_precision_does_not_change_inference_precision(self):
        from sentiment_lab.training import settings
        cfg, base = settings(ROOT / "configs/training/llava_lora.json")
        self.assertEqual(cfg["training_dtype"], "bfloat16")
        self.assertEqual(base["dtype"], "float16")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch needed for CPU tensor mask checks")
class MaskTests(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "image.png"
        Image.new("RGB", (4, 4)).save(self.image)
        class Processor:
            broken = False
            def __init__(self):
                self.tokenizer = self
            def apply_chat_template(self, messages, **kwargs):
                prefix = "<expanded-image>user:post\nassistant:"
                return prefix if len(messages) == 1 else prefix + messages[-1]["content"][0]["text"] + "~"
            def __call__(self, text, **kwargs):
                value = text[0]
                if self.broken and value.endswith("~"):
                    value = "BAD" + value
                ids = torch.tensor([[ord(c) for c in value]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.ones(1, 3, 4, 4)}
            def decode(self, ids, **kwargs):
                return "".join(chr(i) for i in ids if i != ord("~"))
        self.processor = Processor()
        self.row = {"id": "test", "image": str(self.image), "prompt": "post", "label": "positive"}

    def test_answer_only_mask_with_real_cpu_tensors(self):
        import torch
        collator = AnswerOnlyCollator(self.processor, 4096)
        batch = collator([self.row])
        n = collator.last_check["prompt_tokens"]
        self.assertTrue(torch.all(batch["labels"][:, :n] == -100))
        self.assertTrue(torch.equal(batch["labels"][:, n:], batch["input_ids"][:, n:]))
        self.assertEqual(collator.last_check["target"], "positive")
        self.assertIn("pixel_values", batch)

    def test_reject_prefix_mismatch(self):
        self.processor.broken = True
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            AnswerOnlyCollator(self.processor, 4096)([self.row])

    def test_reject_long_sequences_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            AnswerOnlyCollator(self.processor, 2)([self.row])

    def test_reject_unimplemented_visual_batching(self):
        with self.assertRaisesRegex(ValueError, "microbatch"):
            AnswerOnlyCollator(self.processor, 4096)([self.row, self.row])

    def test_gradient_report_distinguishes_missing_zero_nonfinite_and_valid(self):
        import torch
        model = torch.nn.Linear(2, 2, bias=False)
        self.assertEqual(gradient_report(model)["status"], "missing")
        model.weight.grad = torch.zeros_like(model.weight)
        self.assertEqual(gradient_report(model)["status"], "all_zero")
        for value in (float("nan"), float("inf")):
            model.weight.grad = torch.full_like(model.weight, value)
            report = gradient_report(model)
            self.assertEqual(report["status"], "nonfinite")
            self.assertEqual(report["nonfinite_examples"], ["weight"])
        model.weight.grad = torch.ones_like(model.weight)
        report = gradient_report(model)
        self.assertEqual(report["status"], "finite_nonzero")
        self.assertEqual(report["finite_max_abs"], 1.0)

    def test_gradient_report_ignores_frozen_parameters_and_allows_partial_zero(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        for parameter in model[0].parameters():
            parameter.requires_grad_(False)
        model[1].weight.grad = torch.ones_like(model[1].weight)
        model[1].bias.grad = torch.zeros_like(model[1].bias)
        report = gradient_report(model)
        self.assertEqual(report["status"], "finite_nonzero")
        self.assertEqual(report["trainable_tensors"], 2)
        self.assertEqual(report["nonzero_tensors"], 1)
