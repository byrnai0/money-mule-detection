___

Rather than just knowing:

1,501 false positives
338 false negatives

For the false positives, we want to know:

> What legitimate account patterns look suspicious to the model?

For the false negatives:

> What illicit accounts does the model fail to recognize?

___

Created `error_analysis.py`. Run it.

### Output:

| Feature              |    TP |        FP |        FN |     TN |
| -------------------- | ----: | --------: | --------: | -----: |
| Log incoming txns    |  2.33 |  **2.50** |      2.43 |   1.58 |
| Log outgoing txns    |  2.00 |      1.74 |      1.98 |   1.48 |
| Log unique senders   |  1.54 |  **1.76** |  **1.31** |   0.88 |
| Log unique receivers |  1.28 |      1.13 |      1.15 |   0.93 |
| Reciprocity          | 0.031 | **0.056** |     0.003 | 0.0002 |
| Burst ratio          |  1.77 |  **1.89** |      1.82 |   1.47 |
| Concentration        | 0.293 |     0.297 | **0.412** |  0.533 |
___

Dayum alright so heres the problem. the model is not just randomly hallucinating legitimate accounts as suspicious.

It's finding legitimate accounts that **actually exhibit network behavior similar to suspicious accounts**.

> The false-positive group exhibits elevated transaction activity, counterparty breadth, reciprocity and burst behavior relative to true negatives, suggesting that some legitimate high-activity accounts are structurally similar to illicit accounts.

And look at reciprocity:

```
FN = 0.0027
TP = 0.0310
FP = 0.0562
```

So the false negatives are overwhelmingly lacking reciprocal relationships.

That suggests some illicit accounts are **structurally simple** rather than obvious hotspots.

The legitimate accounts that I'm correctly identifying as legitimate often have **high concentration** around a smaller number of counterparties.

The suspicious ngas tend to be **more spread out**.

That's actually consistent with the idea that mule activity can involve broader counterparty interaction rather than a normal account repeatedly dealing with the same small set of entities.

# Changes NEEDED at Phase 4!:

My current features are already capturing:

```
activity
network breadth
burstiness
reciprocity
concentration
```

and the model is clearly using those patterns.

But the false negatives suggest we could eventually benefit from features that describe **small-network suspicious behavior**, rather than just high-volume activity.

