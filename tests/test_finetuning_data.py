"""Synthetic inputs: no real training labels, weights or API calls."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_finetuning", Path(__file__).resolve().parents[1] / "scripts/prepare_finetuning.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class FinetuningDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.frozen = self.root / "source", self.root / "frozen"
        (self.source / "data").mkdir(parents=True)
        self.frozen.mkdir()
        labels = ["ID\ttext,image"]
        for i in range(1, 13):
            label = data.LABELS[(i - 1) // 4]
            labels.append(f"{i}\t{label},{label}")
            (self.source / "data" / f"{i}.jpg").write_bytes(f"image-{i}".encode())
            (self.source / "data" / f"{i}.txt").write_text(f"unique post {i}", encoding="utf-8")
        (self.source / "labelResultAll.txt").write_text("\n".join(labels), encoding="utf-8")
        (self.frozen / "manifest.sha256.json").write_text("{}", encoding="utf-8")
        (self.frozen / "annotations.jsonl").write_text("", encoding="utf-8")
        (self.frozen / "excluded.jsonl").write_text("", encoding="utf-8")
        self.packet = self.root / "packet"
        data.prepare(self.source, self.frozen, self.packet, per_class=3)
        self.rows = data.read_jsonl(self.packet / "annotations.jsonl")
        self.review = {"annotations_sha256": data.file_hash(self.packet / "annotations.jsonl"),
            "reviews": [{"id": r["id"], "label": r["source_text_label"], "reviewer": "synthetic",
                         "reviewed_at": "2026-10-31T00:00:00Z"} for r in self.rows]}
        self.review_file = self.root / "reviews.json"

    def export(self):
        self.review_file.write_text(json.dumps(self.review), encoding="utf-8")
        return data.export(self.packet, self.review_file, self.frozen, self.root / "sft")

    def test_export_disjoint_and_supervision(self):
        self.assertTrue(all(r["human_final_label"] is None for r in self.rows))
        self.assertEqual(self.export(), {"train": 6, "val": 3})
        manifest = data.read_json(self.root / "sft/split_manifest.json")
        groups = {s: {r["id"] for r in manifest["samples"] if r["split"] == s} for s in ["train", "val"]}
        self.assertFalse(groups["train"] & groups["val"])
        for r in data.read_json(self.root / "sft/train.json"):
            self.assertEqual(r["instruction"].count("<image>"), 1)
            self.assertIn(r["output"], data.LABELS)
            self.assertTrue((self.root / "sft" / r["images"][0]).exists())

    def test_reject_incomplete_review(self):
        self.review["reviews"].pop()
        with self.assertRaisesRegex(ValueError, "Every candidate"):
            self.export()
        self.assertFalse((self.root / "sft").exists())

    def test_reject_version_mismatch(self):
        self.review["annotations_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            self.export()

    def test_reject_image_tampering(self):
        (self.packet / self.rows[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.export()

    def test_reject_eval_leakage(self):
        (self.frozen / "excluded.jsonl").write_text(json.dumps(self.rows[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlap"):
            self.export()

    def test_reject_duplicate_reviews(self):
        self.review["reviews"].append(self.review["reviews"][0])
        with self.assertRaisesRegex(ValueError, "Every candidate"):
            self.export()

    def test_no_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_normalised_text_leakage(self):
        row = {**self.rows[0], "source_id": "other", "text": "  " + self.rows[0]["text"].upper() + "  ", "image_sha256": "other"}
        (self.frozen / "annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlap"):
            self.export()

    def test_source_policy_preserves_real_provenance_and_stratifies(self):
        result = data.export(self.packet, None, self.frozen, self.root / "sft",
                             label_policy="source-agreement", source_labels=self.source / "labelResultAll.txt")
        self.assertEqual(result, {"train": 6, "val": 3})
        manifest = data.read_json(self.root / "sft/split_manifest.json")
        self.assertFalse(manifest["human_joint_reviewed"])
        self.assertIsNone(manifest["review_sha256"])
        self.assertTrue(all(r["human_review_status"] == "not_reviewed" and "review" not in r
                            and r["human_final_label"] is None for r in manifest["samples"]))
        for split in ("train", "val"):
            self.assertEqual(set(manifest["counts"][split]), set(data.LABELS))

    def test_source_policy_requires_unchanged_original_label_file(self):
        source = self.source / "labelResultAll.txt"
        source.write_text("ID\ttext,image\n1\tnegative,positive\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source label"):
            data.export(self.packet, None, self.frozen, self.root / "sft",
                        label_policy="source-agreement", source_labels=source)
        self.assertFalse((self.root / "sft").exists())

    def test_source_policy_rejects_non_agreement(self):
        source = self.source / "labelResultAll.txt"
        source.write_text(source.read_text().replace("positive,positive", "negative,positive"), encoding="utf-8")
        metadata = data.read_json(self.packet / "metadata.json")
        metadata["source_labels_sha256"] = data.file_hash(source)
        data.write_json(self.packet / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "agreement mismatch"):
            data.export(self.packet, None, self.frozen, self.root / "sft",
                        label_policy="source-agreement", source_labels=source)


if __name__ == "__main__":
    unittest.main()
