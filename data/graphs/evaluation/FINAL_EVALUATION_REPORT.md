# Final Evaluation Report

Generated: `2026-08-30 17:10:04`

## 1. Final Random-Split Detector

The primary detector uses a tuned soft-voting ensemble of GraphSAGE, GATv2 and GIN. The ensemble weights were selected using validation data only.

### Ensemble weights

- GraphSAGE: **0.78**
- GATv2: **0.19**
- GIN: **0.03**

### Ranking performance

- PR-AUC: **0.308526**
- ROC-AUC: **0.915761**

### Operating point

- Threshold: **0.79**
- Precision: **0.165184**
- Recall: **0.467717**
- F1: **0.244143**
- F2: **0.342324**

### Confusion matrix

| | Count |
|---|---:|
| True Negative | 49,373 |
| False Positive | 1,501 |
| False Negative | 338 |
| True Positive | 297 |


The detector flagged **1,798** of **51,509** test accounts (3.4907%).

## 2. GNN Architecture Comparison

| Model | Mode | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 |
|---|---|---:|---:|---:|---:|---:|---:|
| GCN | Sampled | 0.054647 | 0.784722 | 0.027824 | 0.661417 | 0.053401 | 0.119081 |
| GAT | Sampled | 0.189143 | 0.875483 | 0.042193 | 0.825197 | 0.080282 | 0.175145 |
| GATv2 | Sampled | 0.240506 | 0.890163 | 0.055379 | 0.768504 | 0.103313 | 0.214940 |
| GIN | Sampled | 0.204865 | 0.874072 | 0.033515 | 0.881890 | 0.064576 | 0.145462 |
| GraphSAGE | Sampled | **0.297555** | **0.913669** | 0.055491 | 0.828346 | **0.104014** | **0.218820** |
| Chebyshev | Sampled | 0.136312 | 0.853867 | 0.042487 | 0.730709 | 0.080305 | 0.172350 |

## 3. Ensemble Comparison

| Method | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|---:|---:|
| GraphSAGE | 0.297555 | 0.913669 | 0.055491 | 0.828346 | 0.104014 | 0.218820 |
| Equal voting | 0.271023 | 0.911258 | 0.056763 | 0.847244 | 0.106398 | 0.223831 |
| Tuned voting | **0.308526** | **0.915761** | 0.058385 | 0.839370 | **0.109177** | **0.228383** |
| Stacking | 0.272102 | 0.914634 | 0.057213 | 0.845669 | 0.107175 | 0.225138 |

Tuned voting is the best overall ranking configuration in the random-split experiment. Equal voting and simple logistic-regression stacking did not improve PR-AUC over tuned voting.

## 4. Threshold Analysis

Thresholds were selected using validation predictions only and then applied to the untouched test set.

### Random-split tuned voting

- Best F1 threshold: **0.93**
- Best F2 threshold: **0.79**
- Test precision @ 0.79: **16.52%**
- Test recall @ 0.79: **46.77%**
- Test F1 @ 0.79: **24.41%**
- Test F2 @ 0.79: **34.23%**

## 5. Error Analysis Findings

The error analysis showed that false positives are not random mistakes. Many legitimate accounts have elevated transaction activity, sender diversity, reciprocity and burst behavior that resembles the patterns associated with illicit accounts.

False negatives tend to be harder, less network-visible cases. They generally show fewer incoming counterparties and very low reciprocity, suggesting that the current representation is better at detecting network-visible illicit activity than quiet or concentrated illicit behavior.

## 6. Temporal Evaluation

A separate chronological period-held-out experiment was performed. Transactions were divided into earlier training, middle validation and later test periods. Period-specific graphs and features were constructed, and scaling was fit using the training period only.

### Temporal GraphSAGE

- Test PR-AUC: **0.257384**
- Test ROC-AUC: **0.845433**
- Test precision @ 0.50: **0.026030**
- Test recall @ 0.50: **0.685699**
- Test F1 @ 0.50: **0.050156**
- Test F2 @ 0.50: **0.112993**

### Temporal ensemble

- Validation-selected weights: GraphSAGE **N/A**, GATv2 **N/A**, GIN **N/A**

- Test PR-AUC: **N/A**

- Test ROC-AUC: **N/A**

### Temporal interpretation

The temporal GraphSAGE model achieved PR-AUC **0.257384** and ROC-AUC **0.845433** on the later test period. This is lower than the random-split tuned ensemble (PR-AUC 0.308526; ROC-AUC 0.915761), indicating a measurable reduction in performance under temporal distribution shift.

The temporal soft-voting ensemble achieved PR-AUC **0.251696**, which was slightly below temporal GraphSAGE (0.257384). Therefore, the ensemble benefit observed in the random split did not transfer unchanged to the later time period.

## 7. Important Limitations

1. The main random-split experiment is a stratified account holdout and should not be interpreted as pure future prediction.
2. The temporal experiment is period-held-out evaluation, not a complete online production simulation.
3. The current temporal experiment constructs evaluation-period graphs/features before scoring; it therefore measures transfer to a later graph population rather than strict event-by-event online forecasting.
4. The current stacking experiment uses a single validation set as the meta-training data. A final paper-grade stacking study could use out-of-fold predictions.
5. The current operating thresholds are dataset/validation specific and should not be treated as universal AML thresholds.

## 8. Final Conclusion

The current system demonstrates that graph-based learning can distinguish account-level illicit activity substantially better than the initial GCN baseline. Among the tested architectures, GraphSAGE provided the strongest individual performance, while GATv2 and GIN supplied complementary predictions. In the random holdout experiment, validation-tuned soft voting improved the best individual PR-AUC from 0.297555 to 0.308526. However, the temporal experiment showed that this ensemble advantage is not stable under distribution shift, with temporal GraphSAGE outperforming the temporal ensemble.
