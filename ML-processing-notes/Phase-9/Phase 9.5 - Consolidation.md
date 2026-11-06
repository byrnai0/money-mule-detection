___

Let's consolidate Phase 9 properly so you have a defensible record of what worked, what didn't.

We'll keep the original ensemble as the current best model, preserve each experiment's actual results, and avoid claiming that quantum-inspired or CP-based methods improved detection when the evidence doesn't show that.

**Phase 9 experiment summary**

| Experiment                      | Result                                                                                                             | Conclusion                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| Original ensemble               | Test PR-AUC 0.3060, ROC-AUC 0.9147, F2 0.3496                                                                      | Current benchmark                                         |
| QUBO weight optimization        | Brier-score-selected weights differed from PR-AUC-tuned weights                                                    | No demonstrated advantage over the existing ensemble      |
| QIEA feature selection          | Selected a 12-feature subset in the pilot; hybrid variant did not beat the full-feature baseline on test PR-AUC/F2 | Feature selection remains exploratory                     |
| CP-inspired GNN                 | Test PR-AUC 0.2814, ROC-AUC 0.8966                                                                                 | Below baseline                                            |
| CP-style multiplicative pooling | Test PR-AUC 0.1724, ROC-AUC 0.8755, F2 0.2652                                                                      | Below baseline, even after validation threshold selection |

The QUBO and QIEA rows are qualitative summaries of the results already recorded in our project context; before final submission, we should retrieve their saved artifacts to fill in exact metrics rather than guess at missing values.

___

# Phase 9.1 — QUBO ensemble optimization

The QUBO optimizer used simulated annealing to minimize the validation Brier score, with nonnegative ensemble weights in 0.01 increments that sum to 1.

|Model|Existing weight|QUBO weight|
|---|---|---|
|GraphSAGE|0.78|0.68|
|GATv2|0.19|0.32|
|GIN|0.03|0.00|

The optimizer shifted weight from GraphSAGE to GATv2 and removed GIN from the ensemble.

### Test-set comparison

|Metric|Existing ensemble|QUBO ensemble|Better result|
|---|---|---|---|
|Brier score ↓|0.114922|0.114019|QUBO|
|PR-AUC ↑|0.308526|0.307865|Existing|
|ROC-AUC ↑|0.915761|0.914909|Existing|
|Precision at 0.5 ↑|0.05839|0.05963|QUBO|
|Recall at 0.5 ↑|0.83937|0.83150|Existing|
|F2 at 0.5 ↑|0.22838|0.23170|QUBO|
|Flagged accounts ↓|9,129|8,854|QUBO, fewer alerts|

Source: `qubo_ensemble_report.json`. Arrows indicate the preferred direction for each metric.

Interpretation: QUBO improved probability calibration as measured by Brier score and made slightly fewer alerts at the fixed 0.5 threshold. However, it had slightly lower PR-AUC, ROC-AUC, and recall. Therefore, it is not accurate to claim that QUBO improved overall fraud-detection performance. It produced a different trade-off between calibration, precision, and recall.

One important detail: these metrics use a fixed threshold of 0.5. They should not be directly compared with the baseline results you previously reported using a validation-selected threshold of 0.82.

### QUBO robustness

The robustness report also gives us something useful:

- With penalty values of `0.01`, `0.1`, and `1.0`, the optimizer consistently favored approximately 62–68% GraphSAGE and 32–38% GATv2, with zero GIN weight.
- At penalty `0.001`, the results were much more variable across seeds, including a solution with 56% GATv2.
- All listed solutions were feasible and had weight units summing to 100.

This supports the claim that the optimizer often favors a two-model GraphSAGE/GATv2 mixture under the tested settings. It does not establish that those weights are universally optimal or that a quantum computer would perform better. The experiment used classical simulated annealing.

___
# Phase 9B — QIEA feature selection

The QIEA screening pilot used a class-weighted logistic regression model to evaluate feature subsets, optimizing validation PR-AUC on a sample of 100,000 training records and 30,000 validation records.

|Pilot metric|All features|QIEA-selected features|
|---|---|---|
|Number of features|26|12|
|Validation PR-AUC|0.07406|0.08091|
|Validation F2|0.14174|0.13303|

The 12-feature subset improved pilot PR-AUC by approximately 9.25% relative, but its F2 score decreased by approximately 6.15%.

That means the selected subset performed better at ranking suspicious accounts in this pilot, but did not improve the F2 score at the pilot's chosen operating threshold.

The 12 selected features cover several useful types of behavior:

- Account activity: transaction count, active days, incoming active days.
- Network behavior: unique senders.
- Transaction diversity: payment-format and currency diversity.
- Amount patterns: outgoing transaction mean, standard deviation, maximum, and currency deviation statistics.

These are plausible signals for identifying unusual account activity. However, the pilot report explicitly warns that it is a screening experiment, not the final GNN evaluation.

Important: the final conclusion must come from retraining and testing the GNN using the selected features. We already have an indication from our earlier experiments that the QIEA-based hybrid ensemble did not beat the full-feature baseline on test PR-AUC and F2, so we should not present the pilot's improvement as proof that QIEA improves the final system.

___

# Consolidated Phase 9 results so far

Here is the current comparison using the strongest available test-set evidence.

|Experiment|Test PR-AUC ↑|Test ROC-AUC ↑|Test F2 ↑|Flagged accounts|
|---|---|---|---|---|
|Original ensemble|0.3060|0.9147|0.3496|1,579|
|QUBO ensemble, threshold 0.5|0.3079*|0.9149*|0.2317*|8,854|
|QIEA hybrid ensemble, threshold 0.85|0.3052|0.9148|0.3424|1,169|
|CP-inspired GNN|0.2814|0.8966|0.3297|1,767|
|CP-style pooling GNN|0.1724|0.8755|0.2652|1,570|

*QUBO metrics use threshold 0.5, unlike the validation-selected thresholds used for the other rows. The QUBO row is therefore not a threshold-matched comparison.

The original baseline's metrics are from your earlier evaluation. The QIEA result comes from the saved prediction archive you just evaluated.

## What QIEA actually achieved

At threshold `0.85`, your confusion matrix is:

|Outcome|Accounts|
|---|---|
|True positives|254|
|False positives|915|
|True negatives|49,959|
|False negatives|381|

That means:

- Precision: 21.73% — roughly 22 of every 100 flagged accounts are positive.
- Recall: 40.00% — the model identifies 254 of the 635 positive accounts.
- F2: 0.3424 — close to, but below, the original baseline's 0.3496.
- Alerts: 1,169 — 410 fewer than the baseline's 1,579.

This is a potentially useful operational trade-off: QIEA flags fewer accounts and has higher precision than the original baseline's reported precision of 18.24%, but it misses more positive accounts. Since the baseline and QIEA thresholds were selected separately, treat this as a comparison of their chosen operating points, not proof that QIEA alone caused the difference.

___

# Final Phase 9 verdict

**QUBO — completed experimental optimization**

Demonstrated feasible ensemble-weight optimization and reasonable robustness across tested seeds and penalty values. It improved Brier score but did not improve overall ranking performance.

**QIEA — completed feature-selection experiment**

Reduced the feature set from 26 to 12 in the pilot. The final hybrid ensemble reached PR-AUC 0.3052 and F2 0.3424, slightly below the baseline on those metrics, but used fewer alerts at its selected threshold.

**CP-inspired GNN and CP pooling — explored, no improvement**

Both performed below the original ensemble. Retain these results as negative findings rather than claiming a successful replacement.

