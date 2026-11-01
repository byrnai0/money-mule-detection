___

## What Phase 7A will verify

We'll calculate for the final test set:

```
PR-AUC
ROC-AUC

Precision
Recall
F1
F2

True Positives
True Negatives
False Positives
False Negatives

Number of accounts flagged
Percentage of accounts flagged
```

For our test set, the positive prevalence should be around 1.23%.

Created `evaluate_final.py` file. Run it.

___

### Why we're doing this even though we already have the numbers?

Because Phase 6 gave us results during experimentation.

Phase 7 should create a **clean, standalone evaluation artifact** that someone can run after pulling the repository.

Our current `training_graph.pt` was created from the full graph, and our current feature matrix was also derived from the entire time span. Therefore, this Phase 7 result is a **stratified random account split evaluation**, not yet a strict future-time prediction experiment.

That's okay.

We're going to explicitly label it that way.

Later in Phase 7F we'll do:

```
Chronological split
        ↓
training period
        ↓
future period
        ↓
recompute time-dependent preprocessing
        ↓
train/evaluate
```

That is the harder and more realistic experiment.

___

### Output:

```
Threshold:          0.79
PR-AUC:             0.308526
ROC-AUC:            0.915761
Precision:           0.165184
Recall:              0.467717
F1:                  0.244143
F2:                  0.342324

Confusion matrix:
True Negatives:     49,373
False Positives:    1,501
False Negatives:    338
True Positives:     297

Prediction volume:
Flagged accounts:   1,798
Flagged percentage:  3.4907%
Actual prevalence:   1.2328%
```

So:

- **TP = 297** → correctly caught illicit accounts
- **FN = 338** → missed illicit accounts
- **FP = 1,501** → legitimate accounts incorrectly flagged
- **TN = 49,373** → correctly cleared accounts

The detector flags **3.49% of accounts**, while the actual illicit percentage is only **1.23%**.