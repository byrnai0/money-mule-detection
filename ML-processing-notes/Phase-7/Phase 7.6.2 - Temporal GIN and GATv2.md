___

Now we need to answer:

> Does our carefully tuned random-split ensemble still help when the model is moved through time?

But here's a change from the random experiment:

We **cannot simply reuse the 0.78 / 0.19 / 0.03 weights**.

Those weights were optimized on the random validation set.

For the temporal experiment, the validation period is different, so the ensemble weights must be selected using **temporal validation predictions only**.

The structure becomes:

```
Temporal GraphSAGE ─┐
Temporal GATv2 ─────┼──→ temporal ensemble
Temporal GIN ────────┘
```

### However, there's a catch

We haven't trained temporal GATv2 and temporal GIN.

We only trained temporal GraphSAGE.

So we need to train those two **before** we can do the temporal ensemble.

We'll use the exact same temporal data and sampled setup:

```
Temporal GATv2
Temporal GIN
```

using:

```
batch size = 1024
neighbors = [15, 10]
hidden = 64
dropout = 0.30
same training split
same validation split
```

That keeps the temporal comparison consistent.

___

# IMPORTANT

Create:

```
src/models/train_temporal_gatv2.py
```

no need for another 500-line trainer.

We can reuse the exact structure of `train_temporal_sage.py` and replace:

```
from architectures import GraphSAGEModel
```

with:

```
from architectures import GATv2Model
```

and:

```
model = GraphSAGEModel(
```

with:

```
model = GATv2Model(
```

Everything else stays the same.

Change the output filename to:

```
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_gatv2_model.pt"
)
```

and the title strings from:

```
PHASE 7F-2 — TEMPORAL GRAPHSAGE
```

to:

```
PHASE 7F-3A — TEMPORAL GATV2
```

Then run:

```
python -u .\src\models\train_temporal_gatv2.py
```

## Then temporal GIN

Make the same copy again:

```
src/models/train_temporal_gin.py
```

Change:

```
from architectures import GraphSAGEModel
```

to:

```
from architectures import GINModel
```

and change:

```
model = GraphSAGEModel(
```

to:

```
model = GINModel(
```

Output:

```
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_gin_model.pt"
)
```

Run:

```
python -u .\src\models\train_temporal_gin.py
```

___

## Temporal model comparison

| Model     | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test F2 |
| --------- | ---------- | ----------- | ------------ | -------------- | ----------- | ------- | ------- |
| GraphSAGE | 0.0993     | **0.2574**  | **0.8454**   | 2.60%          | 68.57%      | 0.0502  | 0.1130  |
| GATv2     | 0.1025     | 0.1932      | 0.8227       | 2.22%          | 65.37%      | 0.0428  | 0.0975  |
| GIN       | 0.0415     | 0.2113      | 0.8015       | 1.45%          | **71.88%**  | 0.0283  | 0.0669  |
So **GraphSAGE is still the strongest temporal model overall**. GIN catches the most illicit accounts, but at a much lower precision. GATv2 and GIN runs confirm those values.

