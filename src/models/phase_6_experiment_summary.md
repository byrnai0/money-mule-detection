# Phase 6 Experiment Summary

## Current winner

**Tuned soft voting**

Weights selected on validation data:

- GraphSAGE: **0.78**
- GATv2: **0.19**
- GIN: **0.03**

Test ranking performance:

- **PR-AUC:** 0.308526
- **ROC-AUC:** 0.915761

## Comparison

| Method | PR-AUC | ROC-AUC | Precision @ 0.50 | Recall @ 0.50 | F1 @ 0.50 | F2 @ 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| GraphSAGE | 0.297555 | 0.913669 | 0.055491 | 0.828346 | 0.104014 | 0.218820 |
| GATv2 | 0.240506 | 0.890163 | 0.055379 | 0.768504 | 0.103313 | 0.214940 |
| GIN | 0.204865 | 0.874072 | 0.033515 | 0.881890 | 0.064576 | 0.145462 |
| Equal-weight voting | 0.271023 | 0.911258 | 0.056763 | 0.847244 | 0.106398 | 0.223831 |
| **Tuned soft voting** | **0.308526** | **0.915761** | **0.058385** | **0.839370** | **0.109177** | **0.228383** |
| Stacking | 0.272102 | 0.914634 | 0.057213 | 0.845669 | 0.107175 | 0.225138 |

## Threshold analysis

Thresholds were selected using **validation data only**, then applied to the untouched test set.

### Tuned voting

Best F1 threshold: **0.93**

Best F2 threshold: **0.79**

At the F2 threshold on test:

- Precision: **16.5184%**
- Recall: **46.7717%**
- F1: **24.4143%**
- F2: **34.2324%**
- Positive predictions: **1,798**

### Stacking

Best F1 threshold: **0.95**

Best F2 threshold: **0.90**

At the F2 threshold on test:

- Precision: **19.8091%**
- Recall: **39.2126%**
- F1: **26.3214%**
- F2: **32.7890%**
- Positive predictions: **1,257**

## Conclusions

- Equal-weight voting did **not** beat GraphSAGE on PR-AUC.
- Tuned voting **did** beat GraphSAGE on PR-AUC, ROC-AUC, F1 and F2.
- Stacking worked but did not beat tuned voting on PR-AUC or F2.
- The default 0.50 threshold was too recall-heavy for an operational setting.
- Tuned voting is the current primary ensemble; retain its continuous risk score for later integration.
