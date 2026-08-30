from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
    / "FINAL_EVALUATION_REPORT.md"
)


def fmt(value, digits=6):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def main():

    print("=" * 75)
    print("PHASE 7G — FINAL EVALUATION REPORT")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load random-split final evaluation
    # ---------------------------------------------------------

    print("\n[1/5] Loading random-split evaluation...")

    final_eval_file = (
        EVALUATION_DIR
        / "final_evaluation.csv"
    )

    if not final_eval_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{final_eval_file}\n"
            "Run evaluate_final.py first."
        )

    final_eval = pd.read_csv(
        final_eval_file
    )

    random = final_eval.iloc[0]

    # ---------------------------------------------------------
    # Load threshold analysis
    # ---------------------------------------------------------

    print(
        "[2/5] Loading threshold analysis..."
    )

    threshold_file = (
        EVALUATION_DIR
        / "threshold_analysis.csv"
    )

    threshold = None

    if threshold_file.exists():

        threshold = pd.read_csv(
            threshold_file
        )

    # ---------------------------------------------------------
    # Load temporal results
    # ---------------------------------------------------------

    print(
        "[3/5] Loading temporal evaluation..."
    )

    temporal_model_file = (
        TEMPORAL_DIR
        / "temporal_sage_model.pt"
    )

    temporal_threshold_file = (
        TEMPORAL_DIR
        / "temporal_threshold_results.csv"
    )

    temporal_ensemble_file = (
        TEMPORAL_DIR
        / "temporal_ensemble_results.csv"
    )

    temporal_payload = None

    if temporal_model_file.exists():

        import torch

        temporal_payload = torch.load(
            temporal_model_file,
            map_location="cpu",
            weights_only=False,
        )

    temporal_threshold = None

    if temporal_threshold_file.exists():

        temporal_threshold = pd.read_csv(
            temporal_threshold_file
        )

    temporal_ensemble = None

    if temporal_ensemble_file.exists():

        temporal_ensemble = pd.read_csv(
            temporal_ensemble_file
        )

    # ---------------------------------------------------------
    # Build report
    # ---------------------------------------------------------

    print(
        "[4/5] Building report..."
    )

    generated = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    report = []

    report.append(
        "# Final Evaluation Report\n"
    )

    report.append(
        f"Generated: `{generated}`\n"
    )

    report.append(
        "## 1. Final Random-Split Detector\n"
    )

    report.append(
        "The primary detector uses a tuned soft-voting ensemble "
        "of GraphSAGE, GATv2 and GIN. The ensemble weights were "
        "selected using validation data only.\n"
    )

    report.append(
        "### Ensemble weights\n"
    )

    report.append(
        "- GraphSAGE: **0.78**\n"
        "- GATv2: **0.19**\n"
        "- GIN: **0.03**\n"
    )

    report.append(
        "### Ranking performance\n"
    )

    report.append(
        f"- PR-AUC: **{fmt(random['pr_auc'])}**\n"
        f"- ROC-AUC: **{fmt(random['roc_auc'])}**\n"
    )

    report.append(
        "### Operating point\n"
    )

    report.append(
        f"- Threshold: **{fmt(random['threshold'], 2)}**\n"
        f"- Precision: **{fmt(random['precision'])}**\n"
        f"- Recall: **{fmt(random['recall'])}**\n"
        f"- F1: **{fmt(random['f1'])}**\n"
        f"- F2: **{fmt(random['f2'])}**\n"
    )

    report.append(
        "### Confusion matrix\n"
    )

    report.append(
        "| | Count |\n"
        "|---|---:|\n"
        f"| True Negative | {int(random['true_negatives']):,} |\n"
        f"| False Positive | {int(random['false_positives']):,} |\n"
        f"| False Negative | {int(random['false_negatives']):,} |\n"
        f"| True Positive | {int(random['true_positives']):,} |\n"
    )

    report.append(
        f"\nThe detector flagged **{int(random['positive_predictions']):,}** "
        f"of **{int(random['total_accounts']):,}** test accounts "
        f"({random['flagged_percentage']:.4f}%).\n"
    )

    # ---------------------------------------------------------
    # Random model comparison
    # ---------------------------------------------------------

    report.append(
        "## 2. GNN Architecture Comparison\n"
    )

    report.append(
        "| Model | Mode | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
        "| GCN | Sampled | 0.054647 | 0.784722 | 0.027824 | 0.661417 | 0.053401 | 0.119081 |\n"
        "| GAT | Sampled | 0.189143 | 0.875483 | 0.042193 | 0.825197 | 0.080282 | 0.175145 |\n"
        "| GATv2 | Sampled | 0.240506 | 0.890163 | 0.055379 | 0.768504 | 0.103313 | 0.214940 |\n"
        "| GIN | Sampled | 0.204865 | 0.874072 | 0.033515 | 0.881890 | 0.064576 | 0.145462 |\n"
        "| GraphSAGE | Sampled | **0.297555** | **0.913669** | 0.055491 | 0.828346 | **0.104014** | **0.218820** |\n"
        "| Chebyshev | Sampled | 0.136312 | 0.853867 | 0.042487 | 0.730709 | 0.080305 | 0.172350 |\n"
    )

    # ---------------------------------------------------------
    # Ensemble comparison
    # ---------------------------------------------------------

    report.append(
        "## 3. Ensemble Comparison\n"
    )

    report.append(
        "| Method | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        "| GraphSAGE | 0.297555 | 0.913669 | 0.055491 | 0.828346 | 0.104014 | 0.218820 |\n"
        "| Equal voting | 0.271023 | 0.911258 | 0.056763 | 0.847244 | 0.106398 | 0.223831 |\n"
        "| Tuned voting | **0.308526** | **0.915761** | 0.058385 | 0.839370 | **0.109177** | **0.228383** |\n"
        "| Stacking | 0.272102 | 0.914634 | 0.057213 | 0.845669 | 0.107175 | 0.225138 |\n"
    )

    report.append(
        "Tuned voting is the best overall ranking configuration in "
        "the random-split experiment. Equal voting and simple "
        "logistic-regression stacking did not improve PR-AUC over "
        "tuned voting.\n"
    )

    # ---------------------------------------------------------
    # Threshold analysis
    # ---------------------------------------------------------

    report.append(
        "## 4. Threshold Analysis\n"
    )

    report.append(
        "Thresholds were selected using validation predictions only "
        "and then applied to the untouched test set.\n"
    )

    report.append(
        "### Random-split tuned voting\n"
    )

    report.append(
        "- Best F1 threshold: **0.93**\n"
        "- Best F2 threshold: **0.79**\n"
        "- Test precision @ 0.79: **16.52%**\n"
        "- Test recall @ 0.79: **46.77%**\n"
        "- Test F1 @ 0.79: **24.41%**\n"
        "- Test F2 @ 0.79: **34.23%**\n"
    )

    # ---------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------

    report.append(
        "## 5. Error Analysis Findings\n"
    )

    report.append(
        "The error analysis showed that false positives are not "
        "random mistakes. Many legitimate accounts have elevated "
        "transaction activity, sender diversity, reciprocity and "
        "burst behavior that resembles the patterns associated "
        "with illicit accounts.\n"
    )

    report.append(
        "False negatives tend to be harder, less network-visible "
        "cases. They generally show fewer incoming counterparties "
        "and very low reciprocity, suggesting that the current "
        "representation is better at detecting network-visible "
        "illicit activity than quiet or concentrated illicit behavior.\n"
    )

    # ---------------------------------------------------------
    # Temporal results
    # ---------------------------------------------------------

    report.append(
        "## 6. Temporal Evaluation\n"
    )

    report.append(
        "A separate chronological period-held-out experiment was "
        "performed. Transactions were divided into earlier training, "
        "middle validation and later test periods. Period-specific "
        "graphs and features were constructed, and scaling was fit "
        "using the training period only.\n"
    )

    if temporal_payload is not None:

        val_metrics = temporal_payload.get(
            "validation_metrics"
        )

        test_metrics = temporal_payload.get(
            "test_metrics"
        )

        if test_metrics is not None:

            report.append(
                "### Temporal GraphSAGE\n"
            )

            report.append(
                f"- Test PR-AUC: **{test_metrics['pr_auc']:.6f}**\n"
                f"- Test ROC-AUC: **{test_metrics['roc_auc']:.6f}**\n"
                f"- Test precision @ 0.50: **{test_metrics['precision']:.6f}**\n"
                f"- Test recall @ 0.50: **{test_metrics['recall']:.6f}**\n"
                f"- Test F1 @ 0.50: **{test_metrics['f1']:.6f}**\n"
                f"- Test F2 @ 0.50: **{test_metrics['f2']:.6f}**\n"
            )

    if temporal_ensemble is not None:

        best_ensemble = temporal_ensemble.iloc[0]

        report.append(
            "### Temporal ensemble\n"
        )

        report.append(
            f"- Validation-selected weights: "
            f"GraphSAGE **{best_ensemble['sage_weight'] if 'sage_weight' in best_ensemble else 'N/A'}**, "
            f"GATv2 **{best_ensemble['gatv2_weight'] if 'gatv2_weight' in best_ensemble else 'N/A'}**, "
            f"GIN **{best_ensemble['gin_weight'] if 'gin_weight' in best_ensemble else 'N/A'}**\n"
        )

        report.append(
            f"- Test PR-AUC: **{best_ensemble['test_pr_auc'] if 'test_pr_auc' in best_ensemble else 'N/A'}**\n"
        )

        report.append(
            f"- Test ROC-AUC: **{best_ensemble['test_roc_auc'] if 'test_roc_auc' in best_ensemble else 'N/A'}**\n"
        )

    report.append(
        "### Temporal interpretation\n"
    )

    report.append(
        "The temporal GraphSAGE model achieved PR-AUC **0.257384** "
        "and ROC-AUC **0.845433** on the later test period. This is "
        "lower than the random-split tuned ensemble (PR-AUC 0.308526; "
        "ROC-AUC 0.915761), indicating a measurable reduction in "
        "performance under temporal distribution shift.\n"
    )

    report.append(
        "The temporal soft-voting ensemble achieved PR-AUC **0.251696**, "
        "which was slightly below temporal GraphSAGE (0.257384). "
        "Therefore, the ensemble benefit observed in the random split "
        "did not transfer unchanged to the later time period.\n"
    )

    # ---------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------

    report.append(
        "## 7. Important Limitations\n"
    )

    report.append(
        "1. The main random-split experiment is a stratified account "
        "holdout and should not be interpreted as pure future prediction.\n"
        "2. The temporal experiment is period-held-out evaluation, "
        "not a complete online production simulation.\n"
        "3. The current temporal experiment constructs evaluation-period "
        "graphs/features before scoring; it therefore measures transfer "
        "to a later graph population rather than strict event-by-event "
        "online forecasting.\n"
        "4. The current stacking experiment uses a single validation "
        "set as the meta-training data. A final paper-grade stacking "
        "study could use out-of-fold predictions.\n"
        "5. The current operating thresholds are dataset/validation "
        "specific and should not be treated as universal AML thresholds.\n"
    )

    # ---------------------------------------------------------
    # Final conclusion
    # ---------------------------------------------------------

    report.append(
        "## 8. Final Conclusion\n"
    )

    report.append(
        "The current system demonstrates that graph-based learning can "
        "distinguish account-level illicit activity substantially better "
        "than the initial GCN baseline. Among the tested architectures, "
        "GraphSAGE provided the strongest individual performance, while "
        "GATv2 and GIN supplied complementary predictions. In the random "
        "holdout experiment, validation-tuned soft voting improved the "
        "best individual PR-AUC from 0.297555 to 0.308526. However, the "
        "temporal experiment showed that this ensemble advantage is not "
        "stable under distribution shift, with temporal GraphSAGE "
        "outperforming the temporal ensemble.\n"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "[5/5] Saving final report..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("PHASE 7G COMPLETE")
    print("=" * 75)

    print(
        f"\nReport saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()