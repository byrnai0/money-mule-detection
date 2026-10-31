___

## What Phase 6 was about

Phase 5 gave us three strong GNNs:

- **GraphSAGE** — best individual model
- **GATv2** — second strongest
- **GIN** — strongest recall

Phase 6 was basically: **can combining these models make the detector better?**

---

## 6A — Prediction extraction

We created `ensemble_predictions.py`.

It loaded the saved models and generated probabilities for the same validation/test accounts.

Checks passed:

- Validation: **51,509** predictions per model
- Test: **51,509** predictions per model

Output:

`data/graphs/ensemble_predictions.csv`

---

## 6B — Equal-weight voting

All three models got equal weight:

```text
GraphSAGE = 1/3
GATv2     = 1/3
GIN       = 1/3
```

Test:

- PR-AUC: **0.271023**
- ROC-AUC: **0.911258**
- Precision: **5.68%**
- Recall: **84.72%**
- F1: **0.106398**
- F2: **0.223831**

It did **not** beat GraphSAGE on PR-AUC.

So equal voting was not the final answer.

Output:

`data/graphs/ensemble_voting_results.csv`

---

## 6C — Tuned soft voting

We searched for the best model weights using **validation data only**.

Best weights:

```text
GraphSAGE = 0.78
GATv2     = 0.19
GIN       = 0.03
```

Test at threshold 0.50:

- PR-AUC: **0.308526**
- ROC-AUC: **0.915761**
- Precision: **5.84%**
- Recall: **83.94%**
- F1: **0.109177**
- F2: **0.228383**

This became the **best overall ranking model**.

Output:

`data/graphs/tuned_voting_results.csv`

---

## 6D — Stacking

We also tried:

```text
GraphSAGE probability
GATv2 probability
GIN probability
        ↓
Logistic Regression
        ↓
final probability
```

Test:

- PR-AUC: **0.272102**
- ROC-AUC: **0.914634**
- Precision: **5.72%**
- Recall: **84.57%**
- F1: **0.107175**
- F2: **0.225138**

So stacking worked, but **didn't beat tuned voting**.

Meta-model coefficients:

```text
SAGE  = 0.995
GATv2 = 0.318
GIN   = 0.598
```

Outputs:

- `data/graphs/stacking_results.csv`
- `data/graphs/stacking_meta_model.joblib`

---

## 6E — Threshold selection

Everything above initially used:

```text
probability >= 0.50
```

That was too aggressive for the final operating point.

We searched thresholds on **validation only**.

### Tuned voting

Best F1:

```text
threshold = 0.93
```

Best F2:

```text
threshold = 0.79
```

At **0.79** on the untouched test set:

- Precision: **16.52%**
- Recall: **46.77%**
- F1: **24.41%**
- F2: **34.23%**
- Accounts flagged: **1,798**

### Stacking

Best F1:

```text
threshold = 0.95
```

Best F2:

```text
threshold = 0.90
```

At **0.90** on test:

- Precision: **19.81%**
- Recall: **39.21%**
- F1: **26.32%**
- F2: **32.79%**
- Accounts flagged: **1,257**

---

## What went right

- All three model predictions aligned correctly.
- Tuned voting beat the strongest individual model.
- We kept test data out of weight/threshold selection.
- We now have a continuous risk score plus an operational threshold.
- The three GNNs showed complementary behavior, especially recall vs precision.

## What went wrong

- Equal voting was worse than GraphSAGE on PR-AUC.
- Stacking did not beat tuned voting.
- The default 0.50 threshold produced too many positive predictions / false positives.
- The final threshold is tied to the current validation split and objective; it should not be treated as a universal magic number.

## Current final algorithm

```text
Graph
  +
26 account features
  ↓
GraphSAGE ─┐
GATv2 ─────┼──→ tuned weighted voting
GIN ───────┘
  ↓
risk score
  ↓
threshold selected on validation
  ↓
risk decision
```

Current tuned weights:

```text
GraphSAGE = 0.78
GATv2     = 0.19
GIN       = 0.03
```

Current F2-oriented operating point:

```text
threshold = 0.79
```

**Phase 6 is complete.**
