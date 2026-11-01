___

The workflow is:

```
Temporal validation predictions
          ↓
find best weights
          ↓
freeze weights
          ↓
apply to temporal test predictions
          ↓
evaluate
```

Created `extract_temporal_predictions.py`. Run it.

## Output:

```
[1/4] Loading temporal experiment...
Validation accounts: 248,724
Test accounts:       332,443

[2/4] Extracting temporal model predictions...
SAGE     val=248,724 test=332,443
GATV2    val=248,724 test=332,443
GIN      val=248,724 test=332,443

[3/4] Creating unified temporal prediction table...

[4/4] Running prediction checks...
sage_prob    min=0.000000 max=1.000000
gatv2_prob   min=0.011500 max=0.999999
gin_prob     min=0.000000 max=1.000000
```

Now, create `temporal_ensemble.py`. Run it

## Output:

```
GraphSAGE = 0.18
GATv2     = 0.82
GIN       = 0.00
```

___

### Findings:

### Random split

```
GraphSAGE       = 0.2976
Tuned ensemble  = 0.3085
```

The ensemble **helped**.

### Temporal split

```
GraphSAGE       = 0.2574
Temporal ensemble = 0.2517
```

The ensemble **didn't help**.

That suggests the complementary relationships between the three models are **not stable across time**.

In other words:

> The optimal combination of GNNs learned from one data distribution doesn't necessarily remain optimal when the transaction/network distribution changes.

### Temporal weights

The random experiment gave us:

```
GraphSAGE = 0.78
GATv2     = 0.19
GIN       = 0.03
```

The temporal experiment gave:

```
GraphSAGE = 0.18
GATv2     = 0.82
GIN       = 0.00
```

That's a **massive shift**.

That tells us the models behave differently as the network evolves.

___

For the temporal experiment, freeze:

```
Temporal GraphSAGE
```

as the temporal benchmark.

The temporal ensemble becomes an **experimental negative result**:

> Ensemble weighting did not improve temporal generalization.

Created `temporal_threshold.py`. Run it

## Output:

```
===========================================================================
PHASE 7.6.3 — TEMPORAL THRESHOLD SELECTION
===========================================================================

[1/4] Loading temporal GraphSAGE results...
Validation accounts: 248,724
Test accounts: 332,443

[2/4] Searching validation thresholds...

BEST VALIDATION F1
Threshold: 0.99
Precision: 0.205502
Recall:    0.108919
F1:        0.142377
F2:        0.120220

BEST VALIDATION F2
Threshold: 0.92
Precision: 0.067711
Recall:    0.252144
F1:        0.106754
F2:        0.163225

[3/4] Applying validation-selected F2 threshold to test...

===========================================================================
TEMPORAL TEST — GRAPHSAGE
===========================================================================
THRESHOLD             : 0.92
PRECISION             : 0.164077
RECALL                : 0.383138
F1                    : 0.229760
F2                    : 0.302392
POSITIVE_PREDICTIONS  : 4,376
```

___

Now we can compare the two evaluation settings properly.

### Random account split

Our best detector:

```
Tuned soft voting
```

gave:

```
PR-AUC  = 0.308526
ROC-AUC = 0.915761

At F2 threshold 0.79:
Precision = 16.52%
Recall    = 46.77%
F1        = 24.41%
F2        = 34.23%
```

### Temporal split

Our best temporal model:

```
GraphSAGE
```

gave:

```
PR-AUC  = 0.257384
ROC-AUC = 0.845433

At F2 threshold 0.92:
Precision = 16.41%
Recall    = 38.31%
F1        = 22.98%
F2        = 30.24%
```

So there is a **clear reduction in ranking performance under temporal shift**:

```
PR-AUC:
0.3085 → 0.2574

ROC-AUC:
0.9158 → 0.8454
```

That's not a disaster. In fact, it's exactly the kind of behavior we wanted to measure.

The model generalizes, but **generalization to a later transaction period is harder than random holdout evaluation**.

## And the ensemble behavior is even more interesting

Random experiment:

```
GraphSAGE        PR-AUC = 0.2976
Tuned ensemble   PR-AUC = 0.3085
```

So the ensemble improved the system.

Temporal experiment:

```
Temporal GraphSAGE   PR-AUC = 0.2574
Temporal ensemble    PR-AUC = 0.2517
```

So the temporal ensemble **did not improve** on GraphSAGE.

And remember the temporal weight optimizer selected:

```
GraphSAGE = 0.18
GATv2     = 0.82
GIN       = 0.00
```

yet that ensemble still underperformed temporal GraphSAGE.

That's a very useful project finding:

> **The complementary relationship between the GNNs observed in the random split is not stable under temporal distribution shift.**

