___

For the temporal experiment, we'll use **GraphSAGE as the primary model**.
The pipeline becomes:

```
TRAIN PERIOD
     ↓
GraphSAGE training
     ↓
freeze model
     ↓
VALIDATION PERIOD
     ↓
evaluate
     ↓
TEST PERIOD
     ↓
evaluate
```

### Output:

| Split      |     PR-AUC |    ROC-AUC | Precision |     Recall |         F1 |         F2 |
| ---------- | ---------: | ---------: | --------: | ---------: | ---------: | ---------: |
| Validation |     0.0993 |     0.7615 |     1.64% |     57.12% |     0.0320 |     0.0737 |
| **Test**   | **0.2574** | **0.8454** | **2.60%** | **68.57%** | **0.0502** | **0.1130** |
compared with the earlier random-split experiment:

```
Random-split GraphSAGE
PR-AUC  = 0.2976
ROC-AUC = 0.9137
```

versus:

```
Temporal GraphSAGE
PR-AUC  = 0.2574
ROC-AUC = 0.8454
```

we have a clear performance drop.

# What this means

The model performs better when train/test accounts come from the same overall population, but performance degrades when we move forward in time.

That suggests there is some **temporal distribution shift** in the dataset.

___

## But there's a methodological issue we need to be honest about

The current temporal pipeline builds **separate graphs for each period**:

```
TRAIN GRAPH
VALIDATION GRAPH
TEST GRAPH
```

That is useful for testing temporal transfer, but it doesn't exactly reproduce a real production setup where the system might retain the historical transaction graph and then incrementally add new transactions.

So I would describe this experiment as:

> **Temporal period-held-out evaluation**

rather than:

> **online/future transaction prediction**.

Again, that's not a flaw. It's simply what the experiment actually measures.

