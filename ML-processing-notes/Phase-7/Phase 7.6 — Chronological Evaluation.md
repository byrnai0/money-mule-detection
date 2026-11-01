___

Everything we've done so far is based on the stratified random account split.

That answers:

> "Can the model distinguish illicit accounts when train and test accounts come from the same overall time period?"

But a financial monitoring system should really answer:

> "Can a model trained on earlier activity detect suspicious accounts in later activity?"

So we'll now do:

```
September 1 → earlier period → TRAIN
        ↓
September 2... → later period → TEST
```

But I **cannot just split the existing `account_features_final.csv` by account**, because several features were computed using the entire dataset.

For a proper chronological experiment, we'll need to:

```
Raw cleaned transactions
        ↓
chronological transaction split
        ↓
training-period features
        ↓
test-period representation
        ↓
training labels
        ↓
GNN training
        ↓
future-period evaluation
```
The split will be **70% / 15% / 15% by transaction time**

More importantly, we are **not reusing** the current `training_graph.pt` or `account_features_final.csv` for this experiment. The graph and account features will be rebuilt independently inside each time window. That avoids using future transactions to construct earlier-period features.

This is also consistent with the general PyG approach of using GraphSAGE-style message passing on separate graph structures; GraphSAGE was explicitly designed for inductive representation learning on large graphs.

Created `prepare_temporal_data.py`. Run it

### Output:

```
TRAIN
2022-09-01 → 2022-09-07 14:55
3,554,647 transactions
513,292 accounts
3,857 illicit accounts

VALIDATION
2022-09-07 14:55 → 2022-09-09 03:16
761,590 transactions
248,724 accounts
1,166 illicit accounts

TEST
2022-09-09 03:16 → 2022-09-18 16:18
762,099 transactions
332,443 accounts
1,874 illicit accounts
```

This experiment is **chronological**, but it is not yet a strict "predict the future before seeing its transactions" experiment.

For the validation/test periods, we're constructing:

```
period transactions
      ↓
period graph
      +
period features
      ↓
model evaluation
```

**One observation:**
The account populations aren't stable:

```
Train accounts:       513,292
Validation accounts:  248,724
Test accounts:        332,443
```

There will therefore be:

```
accounts seen earlier
+
accounts appearing later
```

This is particularly interesting for **GraphSAGE**, because its neighborhood-aggregation design is suited to inductive settings.

So our temporal experiment is a good place to test GraphSAGE.

___

