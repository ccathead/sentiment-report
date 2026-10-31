"""Synthetic CPU fixtures only: no MVSA assets, GPU weights, keys or paid API calls."""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentiment_lab.backends import BackendError, DMXBackend, api_payload, clean_trace, response_text
from sentiment_lab.cli import dispatch, parser
from sentiment_lab.core import (LABELS, file_hash, load_config, load_inputs, parse_label,
                                read_json, read_jsonl, render_prompt, write_json)
from sentiment_lab.evaluation import compare_runs, evaluate_run, score
from sentiment_lab.runner import run_experiment


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        for folder in ("inputs", "labels", "images"):
            (self.data / folder).mkdir(parents=True)
        from PIL import Image
        for split in ("dev", "test", "challenge"):
            inputs, labels = [], []
            for i, label in enumerate(LABELS):
                identifier = f"{split}-{i}"
                relative = f"images/{identifier}.png"
                Image.new("RGB", (8, 8), (i * 70, 30, 10)).save(self.data / relative)
                inputs.append({"id": identifier, "image": relative, "text": f"Synthetic post {identifier}"})
                labels.append({"id": identifier, "label": label})
            for folder, rows in (("inputs", inputs), ("labels", labels)):
                (self.data / folder / f"{split}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.refresh_manifest()
        self.config = self.root / "config.json"
        write_json(self.config, {"backend": "hf", "architecture": "llava", "model": "synthetic-test-model",
                                 "revision": "a" * 40, "max_new_tokens": 32})
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("{modality_instruction}\n{post_text_json}", encoding="utf-8")
        self.calls = []
        calls = self.calls

        class Fake:
            info = {"provider": "synthetic-unit-test"}

            def infer(self, prompt, image_path):
                calls.append((prompt, image_path))
                return {"raw_answer": "positive"}

        self.factory = lambda cfg: Fake()

    def refresh_manifest(self):
        write_json(self.data / "manifest.sha256.json",
                   {"files": {p.relative_to(self.data).as_posix(): file_hash(p)
                              for p in self.data.rglob("*") if p.is_file() and p.name != "manifest.sha256.json"}})

    def test_single_evaluation_split_preserves_inputs_labels_and_scoring(self):
        from scripts.consolidate_evaluation import consolidate
        original = load_inputs(self.data, "all")
        from sentiment_lab.core import load_gold
        gold = load_gold(self.data, "all", original)
        merged = self.root / "evaluation-data"
        self.assertEqual(consolidate(self.data, merged), {"eval": 9})
        self.assertEqual(load_inputs(merged, "all"), original)
        self.assertEqual(load_inputs(merged, "eval"), original)
        self.assertEqual(load_gold(merged, "all", original), gold)
        self.assertFalse((merged / "inputs/dev.jsonl").exists())
        out = self.root / "eval-run"
        run_experiment(dataset=merged, config_path=self.config, prompt_path=self.prompt,
                       split="eval", condition="multimodal", out=out, backend_factory=self.factory)
        metrics, rows = evaluate_run(merged, out)
        self.assertEqual(metrics["n"], 9)
        self.assertEqual(metrics["scope"], "full_dataset")
        (merged / "labels/eval.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_gold(merged, "eval", original)

    def run_job(self, **kwargs):
        options = dict(dataset=self.data, config_path=self.config, prompt_path=self.prompt,
                       split="dev", condition="multimodal", out=self.root / "run", backend_factory=self.factory)
        options.update(kwargs)
        return run_experiment(**options)

    def test_inference_never_reads_labels_and_text_has_no_image(self):
        for path in (self.data / "labels").glob("*.jsonl"):
            path.unlink()
        self.run_job(condition="text")
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(image is None for _, image in self.calls))
        self.assertTrue(all("No image is available" in prompt for prompt, _ in self.calls))
        requests = read_jsonl(self.root / "run/requests.jsonl")
        self.assertTrue(all(row["image"] is None for row in requests))

    def test_multimodal_supplies_real_paths(self):
        self.run_job()
        self.assertTrue(all(path.is_file() for _, path in self.calls))

    def test_extra_gold_field_rejected_even_if_hash_matches(self):
        path = self.data / "inputs/dev.jsonl"
        rows = read_jsonl(path)
        rows[0]["label"] = "positive"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "ONLY"):
            load_inputs(self.data, "dev")

    def test_image_tampering_rejected(self):
        (self.data / "images/dev-0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image hash"):
            load_inputs(self.data, "dev")

    def test_path_escape_rejected(self):
        path = self.data / "inputs/dev.jsonl"
        rows = read_jsonl(path)
        rows[0]["image"] = "../private.png"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "inside"):
            load_inputs(self.data, "dev")

    def test_dry_run_does_not_construct_backend_or_score(self):
        self.run_job(dry_run=True, backend_factory=lambda cfg: self.fail("backend constructed"))
        self.assertFalse((self.root / "run/predictions.jsonl").exists())
        with self.assertRaisesRegex(ValueError, "Dry-run"):
            evaluate_run(self.data, self.root / "run")

    def test_resume_skips_flushed_sample_after_interrupt(self):
        calls = []

        class Interrupt:
            info = {}

            def infer(self, prompt, image_path):
                calls.append(prompt)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return {"raw_answer": "positive"}

        with self.assertRaises(KeyboardInterrupt):
            self.run_job(backend_factory=lambda cfg: Interrupt())
        self.assertEqual(len(read_jsonl(self.root / "run/predictions.jsonl")), 1)
        self.run_job(resume=True)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(read_jsonl(self.root / "run/predictions.jsonl")), 3)
        self.assertNotIn("dev-0", self.calls[0][0])

    def test_resume_refuses_modified_prompt(self):
        self.run_job()
        self.prompt.write_text("Changed {post_text_json}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_job(resume=True)

    def test_test_requires_protocol_and_forbids_subset(self):
        with self.assertRaisesRegex(ValueError, "Freeze"):
            self.run_job(split="test")
        with self.assertRaisesRegex(ValueError, "only"):
            self.run_job(split="test", limit=1)

    def test_frozen_protocol_allows_test_and_blocks_changes(self):
        protocol = self.root / "protocol.json"
        dispatch(parser().parse_args(["freeze", "--dataset", str(self.data), "--configs", str(self.config),
                                      "--prompts", str(self.prompt), "--note", "Synthetic test only", "--out", str(protocol)]))
        self.run_job(split="test", protocol=protocol)
        metrics, rows = evaluate_run(self.data, self.root / "run")
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["accuracy"], 1 / 3)
        self.prompt.write_text("Changed {post_text_json}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.run_job(split="challenge", protocol=protocol, out=self.root / "other")

    def test_scoring_detects_gold_tampering(self):
        self.run_job()
        (self.data / "labels/dev.jsonl").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "gold label hash"):
            evaluate_run(self.data, self.root / "run")

    def test_comparison_rejects_different_sample_sets(self):
        self.run_job()
        self.run_job(limit=1, out=self.root / "small")
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_runs(self.data, [self.root / "run", self.root / "small"], self.root / "compare")

    def test_fatal_error_records_one_sample_and_stops(self):
        class Broken:
            info = {}

            def infer(self, prompt, image_path):
                raise BackendError("HTTP 401", fatal=True)

        with self.assertRaises(RuntimeError):
            self.run_job(backend_factory=lambda cfg: Broken())
        metrics, _ = evaluate_run(self.data, self.root / "run")
        self.assertEqual(metrics["invalid_or_missing"], 3)
        self.assertEqual(metrics["missing"], 2)
        self.assertEqual(metrics["accuracy"], 0)

    def test_incomplete_response_cannot_be_counted_as_valid_label(self):
        class Incomplete:
            info = {}

            def infer(self, prompt, image_path):
                return {"raw_answer": "positive", "provider_status": "incomplete"}

        self.run_job(backend_factory=lambda cfg: Incomplete())
        metrics, _ = evaluate_run(self.data, self.root / "run")
        self.assertEqual(metrics["invalid_or_missing"], 3)

    def test_all_inputs_do_not_read_gold(self):
        for path in (self.data / "labels").glob("*.jsonl"):
            path.unlink()
        self.run_job(split="all")
        self.assertEqual(len(self.calls), 9)
        self.assertTrue(all(image.is_file() for _, image in self.calls))

    def test_all_rejects_duplicate_ids_between_parts(self):
        path = self.data / "inputs/challenge.jsonl"
        rows = read_jsonl(path)
        rows[0]["id"] = "dev-0"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_job(split="all")
        self.assertEqual(self.calls, [])

    def test_run_all_loads_model_once_and_scores_every_sample(self):
        args = parser().parse_args(["run-all", "--dataset", str(self.data), "--config", str(self.config),
                                    "--out", str(self.root / "run")])
        with patch("sentiment_lab.backends.HFBackend", side_effect=self.factory) as backend:
            dispatch(args)
        backend.assert_called_once()
        self.assertEqual(len(self.calls), 9)
        self.assertTrue(all(path.is_file() for _, path in self.calls))
        info = read_json(self.root / "run/run.json")
        metrics = read_json(self.root / "run/evaluation/metrics.json")
        self.assertEqual(info["state"], "completed")
        self.assertEqual(info["split"], "all")
        self.assertEqual(info["prompt_name"], "baseline")
        self.assertIsNone(info["protocol_sha256"])
        self.assertEqual(metrics["n"], 9)
        self.assertEqual(metrics["scope"], "full_dataset")
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["missing"], 0)
        self.assertIn("独立保留测试集", (self.root / "run/evaluation/summary.md").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            dispatch(args)

    def test_run_all_stops_on_failure_without_final_score(self):
        class Broken:
            info = {}

            def infer(self, prompt, image_path):
                raise RuntimeError("Synthetic GPU failure")

        args = parser().parse_args(["run-all", "--dataset", str(self.data), "--config", str(self.config),
                                    "--out", str(self.root / "run")])
        with patch("sentiment_lab.backends.HFBackend", return_value=Broken()):
            with self.assertRaisesRegex(RuntimeError, "Run stopped"):
                dispatch(args)
        self.assertEqual(read_json(self.root / "run/run.json")["state"], "failed")
        self.assertFalse((self.root / "run/evaluation").exists())

    def test_run_all_custom_prompt_is_used_and_recorded(self):
        args = parser().parse_args(["run-all", "--dataset", str(self.data), "--config", str(self.config),
                                    "--prompt", str(self.prompt), "--out", str(self.root / "run")])
        with patch("sentiment_lab.backends.HFBackend", side_effect=self.factory):
            dispatch(args)
        info = read_json(self.root / "run/run.json")
        self.assertEqual(info["prompt_name"], "prompt")
        self.assertEqual(info["prompt_sha256"], file_hash(self.prompt))
        self.assertEqual((self.root / "run/prompt_template.txt").read_text(encoding="utf-8"),
                         self.prompt.read_text(encoding="utf-8"))
        self.assertEqual(read_json(self.root / "run/evaluation/metrics.json")["n"], 9)

    def test_run_all_can_resume_an_interrupted_run_and_score(self):
        calls = []

        class Interrupt:
            info = {}

            def infer(self, prompt, image_path):
                calls.append(prompt)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return {"raw_answer": "positive"}

        command = ["run-all", "--dataset", str(self.data), "--config", str(self.config),
                   "--out", str(self.root / "run")]
        with patch("sentiment_lab.backends.HFBackend", return_value=Interrupt()):
            with self.assertRaises(KeyboardInterrupt):
                dispatch(parser().parse_args(command))
        with patch("sentiment_lab.backends.HFBackend", side_effect=self.factory):
            dispatch(parser().parse_args(command + ["--resume"]))
        self.assertEqual(len(self.calls), 8)  # One flushed record was reused, not predicted twice.
        metrics = read_json(self.root / "run/evaluation/metrics.json")
        self.assertEqual(metrics["n"], 9)
        self.assertEqual(metrics["missing"], 0)

    def test_all_scoring_checks_gold_for_every_part(self):
        self.run_job(split="all")
        (self.data / "labels/challenge.jsonl").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "gold label hash"):
            evaluate_run(self.data, self.root / "run")


class MetricsTests(unittest.TestCase):
    def test_exact_parser_rejects_ambiguous_or_explanatory_answers(self):
        self.assertEqual(parse_label(" POSITIVE\n"), "positive")
        self.assertEqual(parse_label('{"label":"neutral"}'), "neutral")
        for value in ("positive or negative", "The label is positive", "positive.", "```positive```", "", None,
                      '{"label":"positive","reason":"nice"}'):
            self.assertIsNone(parse_label(value))

    def test_macro_f1_and_accuracy_keep_invalid_predictions(self):
        gold = {str(i): {"label": label} for i, label in enumerate(LABELS)}
        predictions = [{"id": "2", "prediction": "positive", "status": "ok"},
                       {"id": "0", "prediction": "positive", "status": "ok"},
                       {"id": "1", "prediction": None, "status": "error"}]
        metrics, rows = score(gold, predictions)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 9)
        self.assertEqual(metrics["confusion_matrix"]["negative"]["invalid"], 1)
        self.assertEqual(rows[0]["id"], "0")

    def test_duplicate_unknown_and_missing_ids(self):
        gold = {"a": {"label": "positive"}, "b": {"label": "negative"}}
        pred = {"id": "a", "prediction": "positive", "status": "ok"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score(gold, [pred, pred])
        with self.assertRaisesRegex(ValueError, "unknown"):
            score(gold, [{**pred, "id": "c"}])
        metrics, _ = score(gold, [pred])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["missing"], 1)

    def test_prompt_does_not_interpret_post_as_format_string(self):
        self.assertIn('{"label": "negative"}', render_prompt("{post_text_json}", {"label": "negative"}, "text"))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"backend": "dmx", "model": "synthetic-model", "endpoint": "responses",
                    "base_url": "https://www.dmxapi.cn/v1", "api_key_env": "SYNTHETIC_TEST_KEY",
                    "request_interval_seconds": 0, "max_retries": 2}
        self.env = patch.dict(os.environ, {"SYNTHETIC_TEST_KEY": "test-only-not-a-real-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_payload_contains_real_base64_only_in_image_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "fixture.png"
            image.write_bytes(b"synthetic image bytes")
            for endpoint, field in (("responses", "input"), ("chat/completions", "messages")):
                cfg = {**self.cfg, "endpoint": endpoint}
                body = api_payload(cfg, "Classify", image)
                self.assertIn("data:image/png;base64,", json.dumps(body))
                self.assertEqual(len(body[field][0]["content"]), 2)
                self.assertEqual(len(api_payload(cfg, "Classify", None)[field][0]["content"]), 1)

    def test_response_extracts_answer_not_reasoning(self):
        response = {"output": [{"type": "reasoning", "summary": [{"text": "negative"}]},
                               {"type": "message", "content": [{"type": "output_text", "text": "positive"}]}]}
        self.assertEqual(response_text(response, "responses"), "positive")
        self.assertEqual(response_text({"choices": [{"message": {"content": "neutral"}}]}, "chat/completions"), "neutral")

    def test_retry_transient_error_then_success(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"x-request-id": "synthetic-request"}

        backend = DMXBackend(self.cfg)
        error = urllib.error.HTTPError("https://www.dmxapi.cn/v1/responses", 429, "rate", {}, None)
        success = Response(b'{"output":[]}')
        with patch.object(backend.opener, "open", side_effect=[error, success]) as opened, patch("sentiment_lab.backends.time.sleep"):
            _, attempts = backend.request("POST", "responses", {})
        self.assertEqual(opened.call_count, 2)
        self.assertEqual([a["http_status"] for a in attempts], [429, 200])

    def test_auth_error_is_not_retried(self):
        backend = DMXBackend(self.cfg)
        error = urllib.error.HTTPError("https://www.dmxapi.cn/v1/responses", 401, "secret echo", {}, None)
        with patch.object(backend.opener, "open", side_effect=error) as opened:
            with self.assertRaises(BackendError) as caught:
                backend.request("POST", "responses", {})
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(opened.call_count, 1)
        self.assertNotIn("secret echo", str(caught.exception))

    def test_trace_redacts_key_and_base64(self):
        trace = clean_trace({"text": "echo test-only-key", "url": "data:image/png;base64,YWJj"}, "test-only-key")
        self.assertNotIn("test-only-key", json.dumps(trace))
        self.assertNotIn("YWJj", json.dumps(trace))

    def test_config_cannot_override_inputs_or_include_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            write_json(path, {**self.cfg, "extra_body": {"messages": []}})
            with self.assertRaises(ValueError):
                load_config(path)
            write_json(path, {**self.cfg, "api_key": "never-commit-this"})
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
