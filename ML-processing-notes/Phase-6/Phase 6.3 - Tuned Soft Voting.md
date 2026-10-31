___

> Can we find weights that make the combination better than GraphSAGE alone?

We'll optimize:

```
w_sage
w_gatv2
w_gin
```

subject to:

```
w_sage + w_gatv2 + w_gin = 1
```

and **only validation predictions are allowed to influence those weights**.

Then we apply the chosen weights once to the test predictions.

Created `tuned_voting.py`

### Desired Output:

```
===========================================================================
PHASE 6.3 — TUNED SOFT VOTING
===========================================================================

Searching validation weights...
Weight combinations tested: 5,151

Best validation weights:
GraphSAGE: 0.78
GATv2:     0.19
GIN:       0.03

Best validation results:
PRECISION   : 0.057158
RECALL      : 0.822327
F1          : 0.106887
F2          : 0.223619
PR_AUC      : 0.358850
ROC_AUC     : 0.916273

===========================================================================
TUNED VOTING TEST RESULTS
===========================================================================
PRECISION   : 0.058385
RECALL      : 0.839370
F1          : 0.109177
F2          : 0.228383
PR_AUC      : 0.308526
ROC_AUC     : 0.915761
```

___

## Current ensemble leaderboard:

| Model          |     PR-AUC |    ROC-AUC |  Precision |     Recall |         F1 |         F2 |
| -------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| GraphSAGE      |     0.2976 |     0.9137 |     0.0555 |     0.8283 |     0.1040 |     0.2188 |
| Equal vote     |     0.2710 |     0.9113 |     0.0568 |     0.8472 |     0.1064 |     0.2238 |
| **Tuned vote** | **0.3085** | **0.9158** | **0.0584** | **0.8394** | **0.1092** | **0.2284** |
The validation optimizer chose:

```
GraphSAGE = 0.78
GATv2     = 0.19
GIN       = 0.03
```

So the ensemble is essentially saying:

> GraphSAGE is the strongest base learner, GATv2 adds meaningful complementary information, and GIN contributes only a small amount.

___

The tuned weighted-voting model is currently our best classifier, but we should test whether a meta-classifier can learn a better combination than a fixed weighted average.

The structure becomes:

```
GraphSAGE probability ─┐
GATv2 probability ─────┼──→ Meta-classifier → final probability
GIN probability ───────┘
```

## One important detail

We're currently using `0.5` as the classification threshold. That's fine for architecture/ensemble comparison, but **we're not done with threshold optimization**.

After stacking, Phase 6.5 should compare all the final candidates and then tune the **operating threshold on validation data only**.

That could materially change precision/recall/F1/F2 without changing the underlying model.

So the final pipeline will ultimately be:

```
GNN probabilities
      ↓
ensemble
      ↓
validation threshold selection
      ↓
final risk decision
```

not:
probability > 0.5
by default.

