___

Before combining anything, we need a clean prediction table like:

```
account_id
is_illicit

graphsAGE_prob
gatv2_prob
gin_prob
```

And we need **validation and test predictions separately**.

Basically:
- validation predictions → for choosing ensemble weights/meta-model
- test predictions → **only for final evaluation**

___

Created  `ensemble_predictions.py` cuz the saved sampled-model files contain:

```
model_state_dict
test_probabilities
```
but we didn't save the validation probabilities.

So we need to reconstruct the models and run inference over the validation set.

___

# Why we're doing this before voting

We need exact alignment.

For example, the row:

```
account_id = 123
```

must contain:

```
GraphSAGE probability for account 123
GATv2 probability for account 123
GIN probability for account 123
```

not probabilities in arbitrary loader order.

That's why the script tracks `node_id` and uses the graph's original `account_list`.


___

### Desired Output:

```
===========================================================================
PHASE 6A — ENSEMBLE PREDICTION EXTRACTION
===========================================================================

[1/5] Loading training graph...
Nodes: 515,088

[2/5] Creating validation/test loaders...

[3/5] Extracting model predictions...

  → SAGE
     validation predictions: 51,509
     test predictions: 51,509

  → GATV2
     validation predictions: 51,509
     test predictions: 51,509

  → GIN
     validation predictions: 51,509
     test predictions: 51,509

[4/5] Running prediction alignment checks...
SAGE     val=51,509 test=51,509
GATV2    val=51,509 test=51,509
GIN      val=51,509 test=51,509

[5/5] Saving predictions...

===========================================================================
PHASE 6.1 COMPLETE
===========================================================================

Saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\graphs\ensemble_predictions.csv

Columns:
  - node_id
  - account_id
  - is_illicit
  - sage_val_prob
  - sage_test_prob
  - gatv2_val_prob
  - gatv2_test_prob
  - gin_val_prob
  - gin_test_prob
===========================================================================
```

