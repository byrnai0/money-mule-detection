___

First we establish the simplest ensemble baseline:

$$
P_{\text{ensemble}} = \frac{P_{\text{SAGE}} + P_{\text{GATv2}} + P_{\text{GIN}}}{3}
$$

No optimization yet.

This is important because we'll be able to answer:

> Does combining the models help at all before we start tuning the combination?

Then Phase 6.3 will optimize the weights using **validation only**.

___

Created `ensemble_voting.py`

### Desired Output:

```
===========================================================================
PHASE 6.2 — EQUAL-WEIGHT SOFT VOTING
===========================================================================

[1/4] Loading prediction table...

[2/4] Computing equal-weight validation ensemble...

[3/4] Computing equal-weight test ensemble...

===========================================================================
VALIDATION RESULTS
===========================================================================
PRECISION   : 0.056166
RECALL      : 0.844340
F1          : 0.105325
F2          : 0.221809
PR_AUC      : 0.317152
ROC_AUC     : 0.915613

===========================================================================
TEST RESULTS
===========================================================================
PRECISION   : 0.056763
RECALL      : 0.847244
F1          : 0.106398
F2          : 0.223831
PR_AUC      : 0.271023
ROC_AUC     : 0.911258

[4/4] Saving ensemble probabilities...

Saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\graphs\ensemble_voting_results.csv
```

___

The **PR-AUC** is the first number I'm going to care about.

We'll compare:

```
GraphSAGE  0.2976
GATv2      0.2405
GIN        0.2049
Equal vote ?
```

If equal voting beats GraphSAGE, that's our first evidence that the models are genuinely complementary.

Once we see equal voting, we'll search for:

```
w_sage
w_gatv2
w_gin
```

subject to:

```
w_sage + w_gatv2 + w_gin = 1
```

and select the weights **using validation PR-AUC only**.

Then, and only then, we'll apply the chosen weights to the untouched test predictions.

That gives us a clean:

```
Individual models
        ↓
Equal voting
        ↓
Tuned voting
```

comparison.

After that, we'll do **stacking**.

___

Our best individual model was GraphSAGE:

```
PR-AUC  0.2976
ROC-AUC  0.9137
```

So **equal voting did NOT beat GraphSAGE on PR-AUC**:

```
GraphSAGE    0.2976
Equal vote   0.2710
```

And ROC-AUC is also slightly lower:

```
GraphSAGE    0.9137
Equal vote   0.9113
```

So we should **not claim that simple ensemble voting improved the system**.

Recall improved tho

```
GraphSAGE:
Recall = 82.83%

Equal vote:
Recall = 84.72%
```

and the ensemble's F2:

```
GraphSAGE = 0.2188
Equal vote = 0.2238
```

So the ensemble is slightly better from a **recall-heavy perspective**, while GraphSAGE remains better at overall ranking.

That's exactly why Im moving to **weighted voting** rather than stopping here.

