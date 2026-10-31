___

We already have the three models worth carrying forward:

```
GraphSAGE → strongest overall
GATv2    → strong second
GIN      → strongest recall
```

The goal now is **not to retrain everything**. We already have the models. Phase 6 is about taking their probability outputs and combining them intelligently.

## Phase 6 rundown

```
6.1 → Extract aligned validation/test predictions
6.2 → Equal-weight soft voting baseline
6.3 → Validation-tuned weighted voting
6.4 → Stacking/meta-classifier
6.5 → Compare ensemble vs individual GNNs
6.5 → Freeze the best ensemble
```

This follows the ensemble direction from the Haider paper: combine complementary GNN outputs through tuned soft voting and stacking rather than relying on a single architecture.

### Why we're doing this

Right now:

```
GraphSAGE
→ best overall

GIN
→ catches more positives

GATv2
→ different attention-based representation
```

So instead of asking one model to do everything:

```
GraphSAGE ─┐
GATv2 ─────┼──→ combined risk score
GIN ───────┘
```

we let the models complement one another.

___

Link to other MD files of the same phase:
[[Phase 6 for dummies]]
[[Phase 6.1 - Extract predictions]]
[[Phase 6.2 - Equal-Weight Soft Voting]]
[[Phase 6.3 - Tuned Soft Voting]]
[[Phase 6.4 - Stacking]]
[[Phase 6.5 - Threshold Selection + Final Ensemble Comparison]]
