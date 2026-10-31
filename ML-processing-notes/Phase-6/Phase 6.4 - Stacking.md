___

We'll use:

```
Validation predictions → train meta-classifier
Test predictions      → final evaluation
```

The test set remains untouched while the stacking model is fitted.

Created `stacking.py`

### Desired Output:

```
===========================================================================
PHASE 6.4 — STACKING
===========================================================================

[1/5] Loading prediction table...

[2/5] Preparing validation meta-features...
Validation samples: 51,509

[3/5] Preparing test meta-features...
Test samples: 51,509

[4/5] Training logistic-regression meta-classifier...

===========================================================================
STACKING VALIDATION RESULTS
===========================================================================
PRECISION   : 0.056568
RECALL      : 0.839623
F1          : 0.105994
F2          : 0.222797
PR_AUC      : 0.320547
ROC_AUC     : 0.917371

===========================================================================
STACKING TEST RESULTS
===========================================================================
PRECISION   : 0.057213
RECALL      : 0.845669
F1          : 0.107175
F2          : 0.225138
PR_AUC      : 0.272102
ROC_AUC     : 0.914634

Meta-model coefficients:
SAGE      : 0.995187
GATV2     : 0.318400
GIN       : 0.597883
```

___

stacking did not beat tuned voting

The stacking result is:

```
PR-AUC  = 0.272102
ROC-AUC = 0.914634
F1      = 0.107175
F2      = 0.225138
```

So the current winner remains:

```
Tuned soft voting
PR-AUC = 0.308526
```

## What the stacking coefficients tell us

The meta-model learned:

```
GraphSAGE  = 0.995
GATv2      = 0.318
GIN        = 0.598
```

That reinforces something we already saw experimentally:

```
GraphSAGE = primary signal
GATv2    = useful complementary signal
GIN      = useful but weaker complementary signal
```

Also our stacking meta-model was trained directly on the single validation split.

That's okay for this experiment, but not for the final paper.

A stronger final implementation would use:

```
training data
   ↓
out-of-fold predictions
   ↓
meta-model
   ↓
untouched test
```

That avoids giving the meta-model an overly optimistic view of base-model predictions.

___

Our current best is:

```
GraphSAGE
   +
0.19 GATv2
   +
0.03 GIN
   ↓
Tuned soft voting
```

with:

```
PR-AUC 0.308526
ROC-AUC 0.915761
F1      0.109177
F2      0.228383
```

That's the ensemble I'd currently carry forward.

But **we're not freezing it just yet**.

We now have the validation probabilities for:

```
GraphSAGE
Equal voting
Tuned voting
Stacking
```

So we'll determine the best threshold on the **validation set only**.

We'll evaluate thresholds, for example:

```
0.01
0.02
0.03
...
0.99
```

and identify the operating point under several objectives:

```
maximum F1
maximum F2
high-recall operating point
```

Then we apply the selected threshold **once** to the test probabilities.

That will tell us something much more practical:

> At what risk-score threshold does this AML detector provide a useful precision/recall tradeoff?

