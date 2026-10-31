___

Right now everything has been evaluated at:

```
threshold = 0.50
```

That's useful for apples-to-apples model comparison, but it is not necessarily a sensible operating threshold for a dataset where only about 1.23% of accounts are positive.

For the final AML system, I expect **F2** to be the more relevant primary operating objective because recall is more important than precision when missing a genuinely suspicious account is costly. 

 Created `threshold_selection.py`
### Desired Output:

```
===========================================================================
PHASE 6.5 — THRESHOLD SELECTION
===========================================================================

[1/5] Loading prediction files...

[2/5] Preparing validation data...
Validation accounts: 51,509

[3/5] Searching thresholds...

[4/5] Selecting operating thresholds...

TUNED VOTING — BEST F1
Threshold: 0.93
Precision: 0.523161
Recall:    0.301887
F1:        0.382851
F2:        0.329784

TUNED VOTING — BEST F2
Threshold: 0.79
Precision: 0.173681
Recall:    0.512579
F1:        0.259451
F2:        0.368695

STACKING — BEST F1
Threshold: 0.95
Precision: 0.461078
Recall:    0.242138
F1:        0.317526
F2:        0.267547

STACKING — BEST F2
Threshold: 0.90
Precision: 0.214981
Recall:    0.451258
F1:        0.291223
F2:        0.369941

[5/5] Applying frozen validation thresholds to test...

===========================================================================
FINAL TEST OPERATING POINTS — F2 THRESHOLDS
===========================================================================

TUNED VOTING
THRESHOLD             : 0.79
PRECISION             : 0.165184
RECALL                : 0.467717
F1                    : 0.244143
F2                    : 0.342324
POSITIVE_PREDICTIONS  : 1,798

STACKING
THRESHOLD             : 0.90
PRECISION             : 0.198091
RECALL                : 0.392126
F1                    : 0.263214
F2                    : 0.327890
POSITIVE_PREDICTIONS  : 1,257
```

The key result is that the default `0.5` threshold was **way too aggressive** for an operational AML detector.

### At the validation-selected thresholds:

Tuned voting

Best F1:

```
Threshold 0.93
Precision 52.32%
Recall    30.19%
F1        0.3829
F2        0.3298
```

Best F2:

```
Threshold 0.79
Precision 17.37%
Recall    51.26%
F1        0.2595
F2        0.3687
```

### Stacking

Best F1:

```
Threshold 0.95
Precision 46.11%
Recall    24.21%
F1        0.3175
F2        0.2675
```

Best F2:

```
Threshold 0.90
Precision 21.50%
Recall    45.13%
F1        0.2912
F2        0.3699
```

___
### Once we applied those thresholds **without touching the test set during selection**:

Tuned voting @ 0.79

```
Precision: 16.52%
Recall:    46.77%
F1:        24.41%
F2:        34.23%
```

with:

```
1,798 accounts flagged
```

### Stacking @ 0.90

```
Precision: 19.81%
Recall:    39.21%
F1:        26.32%
F2:        32.79%
```

with:

```
1,257 accounts flagged
```

___

# What this tells us

There are now **two different winners depending on what we optimize**.

### Overall ranking quality

Our tuned voting model is still the strongest:

```
PR-AUC = 0.308526
ROC-AUC = 0.915761
```

### Operational classification at the selected F2 threshold

Tuned voting:

```
F2 = 0.3423
```

Stacking:

```
F2 = 0.3279
```

So **tuned voting remains our current final detector**.

But stacking gives better **precision and F1** at its chosen F2 operating point:

```
Stacking:
19.81% precision
26.32% F1

Tuned voting:
16.52% precision
24.41% F1
```

while tuned voting catches more illicit accounts:

```
46.77% recall vs 39.21%
```

For an AML system, that recall difference is significant.

---

# This also explains the weirdly low F1 I saw earlier

Before threshold optimization, we were doing:

```
probability >= 0.50
```

That produced the earlier:

```
Precision ~5.8%
Recall ~84%
```

So the model wasn't necessarily "bad."

It was simply operating at a **very recall-heavy threshold**.

After moving to 0.79:

```
Precision jumps:
~5.8% → 16.5%

Recall:
~84% → 46.8%
```

That's a completely different operating point.

This is exactly why **PR-AUC + threshold analysis** are more informative than blindly reporting 0.5-threshold F1.

___

# One thing I want to change in our final methodology

I don't think we should write:

> "The final model uses threshold 0.79."

Not yet.

We should define the system as:

```
Model:
Tuned soft-voting ensemble

Risk score:
continuous probability [0,1]

Operating thresholds:
F1-optimized threshold
F2-optimized threshold
```

Then the backend can potentially expose:

```
risk_score = 0.87
```

and separately map it to:

```
LOW / MEDIUM / HIGH
```

later.

That is much more useful than hardcoding one classification boundary into the algorithm.


Main takeaway from this is:
**0.308526 PR-AUC is the best overall ranking result**, while the **0.79 threshold is the current recall-oriented operating point**