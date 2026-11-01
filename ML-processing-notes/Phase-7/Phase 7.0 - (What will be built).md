___

# Phase 7 — Evaluation

Now we're moving from:

> **"Can the model predict?"**

to:

> **"How good, reliable, and useful is the model actually?"**

We'll evaluate the final detector much more thoroughly.

## Phase 7 plan

```
7.1 → Final metric evaluation
7.2 → Confusion matrix + classification report
7.3 → Precision-Recall and ROC curves
7.4 → Risk-score distribution / calibration
7.5 → Error analysis
7.6 → Chronological evaluation
7.7 → Final evaluation report
```

The **chronological evaluation** is especially important. Up to now our main experiment has used a stratified random split. For a financial-detection system, we also need to test whether the model generalizes to later transactions rather than only randomly held-out accounts.

And this is where we'll revisit the temporal leakage issue we discussed earlier.

### The final Phase 7 comparison will eventually look like:

```
Individual GNNs
       ↓
Ensemble
       ↓
Thresholded detector
       ↓
Random split evaluation
       ↓
Chronological evaluation
       ↓
Error analysis
```

We're also going to preserve both:

```
ranking quality:
PR-AUC / ROC-AUC

operating performance:
Precision / Recall / F1 / F2
```

because they're answering different questions.

___

Link to other MD files of the same phase:
[[Phase 7.1 - Final Metric Evaluation]]
[[Phase 7.2 - Confusion Matrix + Error Analysis]]
[[Phase 7.3 - PR and ROC curves]]
[[Phase 7.4 - Risk-Score Distribution]]
[[Phase 7.5 - Deep Error Analysis]]
[[Phase 7.6 — Chronological Evaluation]]
[[Phase 7.6.1 - Temporal GraphSAGE]]
[[Phase 7.6.2 - Temporal GIN and GATv2]]
[[Phase 7.6.3 - Temporal Ensemble]]
[[Phase 7.6.4 - Final Evaluation Report]]
[[Phase 7.7 - Final Evaluation Report]]

