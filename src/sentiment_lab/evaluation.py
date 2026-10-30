from __future__ import annotations

import csv
from pathlib import Path

from .core import LABELS, file_hash, index_rows, load_gold, load_inputs, read_json, read_jsonl, write_json


def score(gold, predictions):
    pred = index_rows(predictions)
    if set(pred) - set(gold):
        raise ValueError("Predictions contain unknown ids")
    matrix = {label: {p: 0 for p in (*LABELS, "invalid")} for label in LABELS}
    evaluated = []
    for identifier, row in gold.items():
        prediction = pred.get(identifier)
        status = prediction.get("status") if prediction else "missing"
        label = prediction.get("prediction") if prediction else None
        valid = status == "ok" and label in LABELS
        label = label if valid else "invalid"
        matrix[row["label"]][label] += 1
        evaluated.append({"id": identifier, "gold": row["label"], "prediction": label,
                          "correct": label == row["label"], "status": status,
                          "raw_answer": prediction.get("raw_answer", "") if prediction else "",
                          "error": prediction.get("error", "") if prediction else "missing prediction"})
    n = len(gold)
    if not n:
        raise ValueError("Cannot score an empty gold set")
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = support - tp  # Invalid/missing predictions remain false negatives.
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    metrics = {"n": n, "correct": sum(r["correct"] for r in evaluated),
               "accuracy": sum(r["correct"] for r in evaluated) / n,
               "macro_f1": sum(r["f1"] for r in per_class.values()) / 3,
               "invalid_or_missing": sum(r["prediction"] == "invalid" for r in evaluated),
               "missing": sum(r["status"] == "missing" for r in evaluated),
               "per_class": per_class, "confusion_matrix": matrix,
               "policy": "All selected gold ids count; invalid/errors/missing are wrong. Macro-F1 averages the three fixed classes."}
    return metrics, evaluated


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(dataset, run):
    run, dataset = Path(run), Path(dataset)
    info = read_json(run / "run.json")
    if info["dry_run"]:
        raise ValueError("Dry-run previews contain no model predictions and cannot be scored")
    if info["dataset_manifest_sha256"] != file_hash(dataset / "manifest.sha256.json"):
        raise ValueError("Evaluation dataset differs from inference snapshot")
    inputs = load_inputs(dataset, info["split"])
    full_gold = load_gold(dataset, info["split"], inputs)
    selected = info["selected_ids"]
    if len(set(selected)) != len(selected) or set(selected) - set(full_gold):
        raise ValueError("Invalid selected ids in run metadata")
    if info["split"] != "dev" and set(selected) != set(full_gold):
        raise ValueError("Test/challenge evaluation must cover the full frozen split")
    predictions = read_jsonl(run / "predictions.jsonl") if (run / "predictions.jsonl").exists() else []
    metrics, rows = score({i: full_gold[i] for i in selected}, predictions)
    by_id = index_rows(inputs)
    for row in rows:
        row.update({"text": by_id[row["id"]]["text"], "image": by_id[row["id"]]["image"],
                    "possible_reason": "", "improvement_idea": ""})
    metrics.update({"split": info["split"], "condition": info["condition"],
                    "prompt": info["prompt_name"], "model": info["config"]["model"],
                    "run_state": info.get("state"), "run_signature": info["signature"],
                    "dataset_manifest_sha256": info["dataset_manifest_sha256"],
                    "scope": "dev_subset" if len(selected) < len(full_gold) else "full_split",
                    "prediction_file_sha256": file_hash(run / "predictions.jsonl") if predictions else None,
                    "observed_inference_seconds": sum(p.get("seconds", 0) for p in predictions),
                    "api_usage": [p["usage"] for p in predictions if p.get("usage")]})
    return metrics, rows


def save_evaluation(out, metrics, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "metrics.json", metrics)
    fields = ["id", "gold", "prediction", "correct", "status", "text", "image", "raw_answer", "error", "possible_reason", "improvement_idea"]
    write_csv(out / "per_sample.csv", rows, fields)
    errors = [r for r in rows if not r["correct"]]
    write_csv(out / "errors.csv", errors, fields)
    write_csv(out / "confusion_matrix.csv", [{"gold": k, **v} for k, v in metrics["confusion_matrix"].items()],
              ["gold", *LABELS, "invalid"])
    (out / "summary.md").write_text(
        f"# 实验评分\n\n模型：`{metrics['model']}`；划分：{metrics['split']}；输入：{metrics['condition']}；"
        f"prompt：{metrics['prompt']}。\n\n"
        f"- 范围：{metrics['scope']}，n={metrics['n']}。\n"
        f"- Accuracy：{metrics['accuracy']:.4f}（{metrics['correct']}/{metrics['n']}）。\n"
        f"- Macro-F1：{metrics['macro_f1']:.4f}。\n"
        f"- 无效/缺失：{metrics['invalid_or_missing']}，全部计错。\n"
        f"- 运行状态：{metrics['run_state']}；未完成运行的缺失项也已计错。\n\n"
        "错误案例见 errors.csv；possible_reason 和 improvement_idea 留空，须结合原图人工分析。"
        "不能把程序自动计分当成已完成人工错误解释。\n", encoding="utf-8")


def compare_runs(dataset, runs, out):
    evaluated = [evaluate_run(dataset, run) for run in runs]
    first_metrics, first_rows = evaluated[0]
    reference = {r["id"]: r["gold"] for r in first_rows}
    for metrics, rows in evaluated[1:]:
        if metrics["split"] != first_metrics["split"] or {r["id"]: r["gold"] for r in rows} != reference:
            raise ValueError("Comparison requires identical split, ids and gold labels")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    table = [{"run": str(run), **metrics} for run, (metrics, _) in zip(runs, evaluated)]
    write_csv(out / "comparison.csv", table,
              ["run", "model", "split", "condition", "prompt", "n", "accuracy", "macro_f1", "invalid_or_missing", "observed_inference_seconds"])
    paired = []
    base = index_rows(first_rows)
    for run, (_, rows) in zip(runs[1:], evaluated[1:]):
        for row in rows:
            old = base[row["id"]]
            change = ("corrected" if not old["correct"] and row["correct"] else
                      "new_error" if old["correct"] and not row["correct"] else
                      "both_correct" if row["correct"] else "both_wrong")
            paired.append({"comparison_run": str(run), "id": row["id"], "gold": row["gold"],
                           "baseline_prediction": old["prediction"], "comparison_prediction": row["prediction"], "change": change})
    write_csv(out / "paired_changes.csv", paired,
              ["comparison_run", "id", "gold", "baseline_prediction", "comparison_prediction", "change"])
    write_json(out / "comparison.json", table)
    return table
