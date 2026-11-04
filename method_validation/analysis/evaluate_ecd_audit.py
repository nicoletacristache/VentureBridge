#!/usr/bin/env python3
"""Evaluate matched ECD predictions against the reference-label file.

See ../PROVENANCE.md for data source, evaluation assumptions, and scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


CLASSES = np.arange(5, dtype=int)
DIMENSIONS = (
    "opportunity_framing",
    "evidence_falsifiability",
    "resource_finance",
    "adaptive_action",
    "stakeholder_ethics",
)
METHOD_LABELS = {
    "rule": "ECD关键词/结构规则",
    "llm": "VentureBridge结构化ECD编码（终止失败时规则回退）",
}
PLOT_METHOD_LABELS = {
    "rule": "Rule baseline",
    "llm": "Structured ECD\n(rule fallback on terminal failure)",
}
SEED = 20261030
N_BOOTSTRAP = 2000
RATE_ESTIMATOR = "jeffreys_posterior_mean_beta_0.5_0.5"
LLM_TERMINAL_FAILURE_POLICY = "deterministic_same_item_rule_fallback_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_information(value: str) -> str | list[Any] | dict[str, Any]:
    """Mirror the scorer's lossless JSON-aware input normalization."""
    value = value.strip()
    if not value:
        return ""
    if value[:1] in {"[", "{"}:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, (list, dict)):
            return parsed
    return value


def current_item_hashes(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Rebuild the scorer's item hashes from the current frozen audit CSV.

    Prediction JSONL files are append-only and may legitimately retain prior
    versions of the same ``audit_id``.  The content hash, not the bare ID,
    determines which version is eligible for the current evaluation.
    """

    items = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "audit_id",
        "scenario_text",
        "decision_text",
        "rationale_text",
        "information_requests",
        "revision_text",
        "target_dimension",
    }
    if not required.issubset(items.columns):
        raise ValueError(f"audit item file lacks: {sorted(required - set(items.columns))}")
    if len(items) != 300 or items["audit_id"].nunique() != 300:
        raise ValueError("the frozen ECD audit item file must contain 300 unique items")
    hashes: dict[str, str] = {}
    targets: dict[str, str] = {}
    for row in items.to_dict(orient="records"):
        audit_id = str(row["audit_id"]).strip()
        target = str(row["target_dimension"]).strip()
        if target not in DIMENSIONS:
            raise ValueError(f"unknown target dimension for {audit_id}: {target!r}")
        context = {
            "event_kind": "main",
            "event_text": str(row["scenario_text"]).strip(),
            "target_dimensions": list(DIMENSIONS),
            "allowed_actions": [
                {
                    "action_code": "AUDIT_RESPONSE",
                    "label": "编码当前回答",
                    "description": "固定审计集的证据编码；不触发任何模拟状态变化。",
                }
            ],
            "response": {
                "decision_text": str(row["decision_text"]).strip(),
                "rationale_text": str(row["rationale_text"]).strip(),
                "information_requests": parse_information(
                    str(row["information_requests"])
                ),
                "revision_text": str(row["revision_text"]).strip(),
            },
        }
        hashes[audit_id] = hash_json(
            {"sample_id": audit_id, "target_dimension": target, "context": context}
        )
        targets[audit_id] = target
    return hashes, targets


def load_jsonl(
    path: Path,
    method: str,
    expected_hashes: dict[str, str],
    expected_targets: dict[str, str],
) -> pd.DataFrame:
    # Historical records with other item hashes may remain in the append log,
    # but the frozen item hash must have exactly one terminal record.  This
    # deliberately rejects outer retry-until-success or result resampling.
    selected: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in expected_hashes:
                continue
            if str(record.get("item_hash", "")) != expected_hashes[sample_id]:
                continue
            if record.get("method") != method:
                raise ValueError(f"{path}:{line_number} has wrong method for {sample_id}")
            selected.setdefault(sample_id, []).append((line_number, record))

    if len(selected) != len(expected_hashes):
        missing = sorted(set(expected_hashes) - set(selected))
        raise ValueError(
            f"{method} has {len(selected)} current-item terminal records; expected "
            f"{len(expected_hashes)} (missing first: {missing[:5]})"
        )
    repeated = sorted(sample_id for sample_id, values in selected.items() if len(values) != 1)
    if repeated:
        raise ValueError(
            f"{method} has repeated current-item records, which would permit resampling: "
            f"{repeated[:5]}"
        )
    config_hashes = {
        str(values[0][1].get("config_hash", "")) for values in selected.values()
    }
    if len(config_hashes) != 1 or not next(iter(config_hashes), ""):
        raise ValueError(f"{method} current-item predictions mix configurations")

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(selected):
        line_number, record = selected[sample_id][0]
        if record.get("target_dimension") != expected_targets[sample_id]:
            raise ValueError(f"{path}:{line_number} target mismatch for {sample_id}")
        status = str(record.get("status", ""))
        if status not in {"success", "failed"}:
            raise ValueError(f"{path}:{line_number} has invalid terminal status {status!r}")
        scores: dict[str, int | float] = {dimension: np.nan for dimension in DIMENSIONS}
        if status == "success":
            output = record.get("raw_json")
            evidence = output.get("evidence") if isinstance(output, dict) else None
            if not isinstance(evidence, list):
                raise ValueError(f"{path}:{line_number} has no evidence array")
            parsed_scores: dict[str, int] = {}
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                dimension = item.get("dimension")
                score = item.get("score")
                if dimension in DIMENSIONS and isinstance(score, int) and 0 <= score <= 4:
                    parsed_scores[str(dimension)] = int(score)
            if set(parsed_scores) != set(DIMENSIONS):
                raise ValueError(f"{path}:{line_number} lacks five unique dimension scores")
            scores.update(parsed_scores)
        elif method == "rule":
            raise ValueError(f"{path}:{line_number} rule baseline has a terminal failure")
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        rows.append(
            {
                "audit_id": sample_id,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "primary_status": status,
                "failure_code": str(error.get("code", "")) if status == "failed" else "",
                "fallback_used": False,
                "prediction_source": method,
                **{f"score_{dimension}": scores[dimension] for dimension in DIMENSIONS},
            }
        )
    return pd.DataFrame(rows)


def apply_llm_terminal_failure_policy(
    llm_prediction: pd.DataFrame, rule_prediction: pd.DataFrame
) -> pd.DataFrame:
    """Fill terminal LLM failures from the frozen same-item rule baseline.

    This policy is deterministic, label-blind, fixed before the final holdout
    run, and does not make another provider call.  The original failed JSONL
    record remains unchanged and is exposed through ``primary_status`` and
    ``failure_code`` in the focal-prediction audit table.
    """

    output = llm_prediction.copy()
    rule_by_id = rule_prediction.set_index("audit_id")
    failed = output["primary_status"].eq("failed")
    for index in output.index[failed]:
        audit_id = str(output.at[index, "audit_id"])
        if audit_id not in rule_by_id.index:
            raise ValueError(f"missing same-item rule fallback for {audit_id}")
        for dimension in DIMENSIONS:
            output.at[index, f"score_{dimension}"] = int(
                rule_by_id.at[audit_id, f"score_{dimension}"]
            )
        output.at[index, "fallback_used"] = True
        output.at[index, "prediction_source"] = LLM_TERMINAL_FAILURE_POLICY
    score_columns = [f"score_{dimension}" for dimension in DIMENSIONS]
    if output[score_columns].isna().any().any():
        raise ValueError("LLM failure policy left one or more dimension scores missing")
    output[score_columns] = output[score_columns].astype(int)
    return output


def focal_predictions(reference: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    joined = reference.merge(prediction, on="audit_id", how="inner", validate="one_to_one")
    joined["predicted_score"] = [
        int(row[f"score_{row['target_dimension']}"]) for _, row in joined.iterrows()
    ]
    return joined[
        [
            "audit_id",
            "target_dimension",
            "reference_score",
            "case_type",
            "method",
            "method_label",
            "predicted_score",
            "primary_status",
            "failure_code",
            "fallback_used",
            "prediction_source",
        ]
    ]


def class_metrics(cm: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    total = int(cm.sum())
    for label in CLASSES:
        tp = int(cm[label, label])
        fn = int(cm[label, :].sum() - tp)
        fp = int(cm[:, label].sum() - tp)
        tn = total - tp - fn - fp
        precision = jeffreys_rate(tp, tp + fp)
        recall = jeffreys_rate(tp, tp + fn)
        specificity = jeffreys_rate(tn, tn + fp)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": int(label),
                "support": int(cm[label, :].sum()),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "rate_estimator": RATE_ESTIMATOR,
            }
        )
    return rows


def jeffreys_rate(successes: int | np.integer, total: int | np.integer) -> float:
    """Posterior mean for a binomial rate under Jeffreys' Beta(.5,.5) prior.

    The audit has only 12 reference cases per dimension-by-level cell.  Raw
    proportions at that granularity can equal zero or one by construction and
    overstate certainty.  The smoothed point estimate is used for all reported
    binomial rates; raw integer counts remain available in every confusion
    matrix.  Undefined rates with a zero denominator remain missing.
    """

    if int(total) <= 0:
        return float("nan")
    return float((int(successes) + 0.5) / (int(total) + 1.0))


def evaluate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    for (method, method_label, dimension), frame in predictions.groupby(
        ["method", "method_label", "target_dimension"], sort=False
    ):
        y_true = frame["reference_score"].to_numpy(int)
        y_pred = frame["predicted_score"].to_numpy(int)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        rows = class_metrics(cm)
        for row in rows:
            per_class.append(
                {
                    "method": method,
                    "method_label": method_label,
                    "dimension": dimension,
                    **row,
                }
            )
        for true_label in CLASSES:
            for predicted_label in CLASSES:
                matrices.append(
                    {
                        "method": method,
                        "method_label": method_label,
                        "dimension": dimension,
                        "reference_score": int(true_label),
                        "predicted_score": int(predicted_label),
                        "count": int(cm[true_label, predicted_label]),
                    }
                )
        recalls = np.array([row["recall"] for row in rows], dtype=float)
        f1s = np.array([row["f1"] for row in rows], dtype=float)
        summaries.append(
            {
                "method": method,
                "method_label": method_label,
                "dimension": dimension,
                "n": int(len(frame)),
                "accuracy": jeffreys_rate(int(np.sum(y_true == y_pred)), len(y_true)),
                "macro_f1": float(np.nanmean(f1s)),
                "balanced_accuracy": float(np.nanmean(recalls)),
                "quadratic_weighted_kappa": float(
                    cohen_kappa_score(y_true, y_pred, labels=CLASSES, weights="quadratic")
                ),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "exact_agreement": jeffreys_rate(
                    int(np.sum(y_true == y_pred)), len(y_true)
                ),
                "within_one_agreement": jeffreys_rate(
                    int(np.sum(np.abs(y_true - y_pred) <= 1)), len(y_true)
                ),
                "rate_estimator": RATE_ESTIMATOR,
            }
        )
    # Pooled rows retain the balanced 60-item contribution of each dimension.
    for (method, method_label), frame in predictions.groupby(["method", "method_label"]):
        y_true = frame["reference_score"].to_numpy(int)
        y_pred = frame["predicted_score"].to_numpy(int)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        rows = class_metrics(cm)
        for row in rows:
            per_class.append(
                {
                    "method": method,
                    "method_label": method_label,
                    "dimension": "ALL",
                    **row,
                }
            )
        for true_label in CLASSES:
            for predicted_label in CLASSES:
                matrices.append(
                    {
                        "method": method,
                        "method_label": method_label,
                        "dimension": "ALL",
                        "reference_score": int(true_label),
                        "predicted_score": int(predicted_label),
                        "count": int(cm[true_label, predicted_label]),
                    }
                )
        summaries.append(
            {
                "method": method,
                "method_label": method_label,
                "dimension": "ALL",
                "n": int(len(frame)),
                "accuracy": jeffreys_rate(int(np.sum(y_true == y_pred)), len(y_true)),
                "macro_f1": float(np.nanmean([row["f1"] for row in rows])),
                "balanced_accuracy": float(np.nanmean([row["recall"] for row in rows])),
                "quadratic_weighted_kappa": float(
                    cohen_kappa_score(y_true, y_pred, labels=CLASSES, weights="quadratic")
                ),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "exact_agreement": jeffreys_rate(
                    int(np.sum(y_true == y_pred)), len(y_true)
                ),
                "within_one_agreement": jeffreys_rate(
                    int(np.sum(np.abs(y_true - y_pred) <= 1)), len(y_true)
                ),
                "rate_estimator": RATE_ESTIMATOR,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(per_class), pd.DataFrame(matrices)


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(predictions["method"].unique())
    if methods != ["llm", "rule"]:
        return pd.DataFrame()
    wide = predictions.pivot(
        index="audit_id", columns="method", values=["reference_score", "predicted_score"]
    )
    truth = wide[("reference_score", "llm")].to_numpy(int)
    llm = wide[("predicted_score", "llm")].to_numpy(int)
    rule = wide[("predicted_score", "rule")].to_numpy(int)
    if not np.array_equal(truth, wide[("reference_score", "rule")].to_numpy(int)):
        raise ValueError("methods were not evaluated against identical references")
    rng = np.random.default_rng(SEED + 77)
    delta_mae = np.empty(N_BOOTSTRAP)
    delta_accuracy = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(truth), len(truth))
        delta_mae[index] = np.mean(np.abs(truth[sample] - rule[sample])) - np.mean(
            np.abs(truth[sample] - llm[sample])
        )
        delta_accuracy[index] = np.mean(llm[sample] == truth[sample]) - np.mean(
            rule[sample] == truth[sample]
        )
    return pd.DataFrame(
        [
            {
                "contrast": "llm_minus_rule",
                "n_items": len(truth),
                "bootstrap_replicates": N_BOOTSTRAP,
                "mae_reduction": float(np.mean(np.abs(truth - rule)) - np.mean(np.abs(truth - llm))),
                "mae_reduction_ci_low": float(np.quantile(delta_mae, 0.025)),
                "mae_reduction_ci_high": float(np.quantile(delta_mae, 0.975)),
                "accuracy_increase": float(np.mean(llm == truth) - np.mean(rule == truth)),
                "accuracy_increase_ci_low": float(np.quantile(delta_accuracy, 0.025)),
                "accuracy_increase_ci_high": float(np.quantile(delta_accuracy, 0.975)),
            }
        ]
    )


def case_type_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize common, boundary, and noisy expression conditions.

    These strata are fixed by the data generator before inference.  They are
    diagnostic error-analysis conditions, not post-hoc subsets selected from
    prediction errors.
    """

    rows: list[dict[str, Any]] = []
    for (method, method_label, case_type), frame in predictions.groupby(
        ["method", "method_label", "case_type"], sort=False
    ):
        y_true = frame["reference_score"].to_numpy(int)
        y_pred = frame["predicted_score"].to_numpy(int)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        per_class = class_metrics(cm)
        rows.append(
            {
                "method": method,
                "method_label": method_label,
                "case_type": case_type,
                "n": int(len(frame)),
                "accuracy": jeffreys_rate(int(np.sum(y_true == y_pred)), len(y_true)),
                "macro_f1": float(np.nanmean([row["f1"] for row in per_class])),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "within_one_agreement": jeffreys_rate(
                    int(np.sum(np.abs(y_true - y_pred) <= 1)), len(y_true)
                ),
                "errors_over_one": int(np.sum(np.abs(y_true - y_pred) > 1)),
                "rate_estimator": RATE_ESTIMATOR,
            }
        )
    return pd.DataFrame(rows)


def plot_matrices(matrices: pd.DataFrame, output: Path) -> None:
    methods = [method for method in ("rule", "llm") if method in set(matrices["method"])]
    fig, axes = plt.subplots(len(methods), len(DIMENSIONS), figsize=(18, 6.7), squeeze=False)
    for row_index, method in enumerate(methods):
        for column_index, dimension in enumerate(DIMENSIONS):
            ax = axes[row_index, column_index]
            part = matrices.loc[
                matrices["method"].eq(method) & matrices["dimension"].eq(dimension)
            ]
            cm = part.pivot(
                index="reference_score", columns="predicted_score", values="count"
            ).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
            image = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=max(1, int(cm.to_numpy().max())))
            for i in CLASSES:
                for j in CLASSES:
                    value = int(cm.loc[i, j])
                    ax.text(j, i, value, ha="center", va="center", fontsize=8)
            ax.set_xticks(CLASSES)
            ax.set_yticks(CLASSES)
            if row_index == len(methods) - 1:
                ax.set_xlabel("Predicted")
            if column_index == 0:
                ax.set_ylabel(f"{PLOT_METHOD_LABELS[method]}\nReference")
            ax.set_title(dimension.replace("_", "\n"), fontsize=9)
    fig.suptitle("Five-level ECD coding confusion matrices")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("method_validation/data/ecd_reference_labels.csv")
    )
    parser.add_argument(
        "--items", type=Path, default=Path("method_validation/data/ecd_audit_items.csv")
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("method_validation/predictions")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("method_validation/results"))
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    required = {"audit_id", "target_dimension", "reference_score", "case_type"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reference file lacks: {sorted(required - set(reference.columns))}")
    if len(reference) != 300 or reference["audit_id"].nunique() != 300:
        raise ValueError("the frozen ECD audit reference must contain 300 unique items")
    expected_hashes, expected_targets = current_item_hashes(args.items)
    if set(reference["audit_id"].astype(str)) != set(expected_hashes):
        raise ValueError("audit item and reference ID sets differ")
    reference_targets = dict(
        zip(reference["audit_id"].astype(str), reference["target_dimension"].astype(str))
    )
    if reference_targets != expected_targets:
        raise ValueError("audit item and reference target dimensions differ")
    rule_prediction = load_jsonl(
        args.predictions_dir / "ecd_audit_rule.jsonl",
        "rule",
        expected_hashes,
        expected_targets,
    )
    llm_primary = load_jsonl(
        args.predictions_dir / "ecd_audit_llm.jsonl",
        "llm",
        expected_hashes,
        expected_targets,
    )
    for method, prediction in (("rule", rule_prediction), ("llm", llm_primary)):
        if len(prediction) != 300:
            raise ValueError(f"{method} has {len(prediction)} terminal predictions; expected 300")
    llm_prediction = apply_llm_terminal_failure_policy(llm_primary, rule_prediction)
    prediction_parts = [
        focal_predictions(reference, rule_prediction),
        focal_predictions(reference, llm_prediction),
    ]
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summaries, per_class, matrices = evaluate(predictions)
    bootstrap = paired_bootstrap(predictions)
    case_types = case_type_metrics(predictions)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_dir / "ecd_audit_focal_predictions.csv", index=False)
    summaries.to_csv(args.results_dir / "ecd_audit_metrics.csv", index=False)
    per_class.to_csv(args.results_dir / "ecd_audit_per_class_metrics.csv", index=False)
    matrices.to_csv(args.results_dir / "ecd_audit_confusion_matrices.csv", index=False)
    bootstrap.to_csv(args.results_dir / "ecd_audit_paired_bootstrap.csv", index=False)
    case_types.to_csv(args.results_dir / "ecd_audit_case_type_metrics.csv", index=False)
    plot_matrices(matrices, args.results_dir / "ecd_audit_confusion_matrices.png")
    summary = {
        "status": "completed",
        "n_items": 300,
        "rate_estimator": RATE_ESTIMATOR,
        "llm_terminal_failure_policy": LLM_TERMINAL_FAILURE_POLICY,
        "llm_terminal_failures": int(llm_primary["primary_status"].eq("failed").sum()),
        "methods": methods_from(summaries),
        "pooled_metrics": summaries.loc[summaries["dimension"].eq("ALL")].to_dict(
            orient="records"
        ),
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "case_type_metrics": case_types.to_dict(orient="records"),
    }
    (args.results_dir / "ecd_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "n_items": 300}, ensure_ascii=False))


def methods_from(summary: pd.DataFrame) -> list[str]:
    return sorted(summary["method"].astype(str).unique())


if __name__ == "__main__":
    main()
