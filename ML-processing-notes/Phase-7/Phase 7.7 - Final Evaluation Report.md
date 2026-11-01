___

The clean version of the story is:

```
Random split
→ Tuned soft voting is best
→ PR-AUC = 0.308526
→ ROC-AUC = 0.915761

Temporal split
→ GraphSAGE is best
→ PR-AUC = 0.257384
→ ROC-AUC = 0.845433

Temporal ensemble
→ did not outperform temporal GraphSAGE
```

And the main detector currently carried forward is:

```
GraphSAGE = 0.78
GATv2     = 0.19
GIN       = 0.03
```

with the random-split F2 operating threshold:

```
0.79
```

while the temporal GraphSAGE experiment selected:

```
0.92
```

That distinction is important: **the model/ensemble and its operating threshold aren't universal constants**.

## Phase 7 is now frozen

From here, we shouldn't casually modify the existing evaluation/model artifacts. They form our benchmark.

The next phase can build on top of this.

And yeah, we now have an actual research result instead of just a collection of scripts:

**the graph models clearly learn useful AML signal, tuned ensemble voting improves the random-split benchmark, but temporal distribution shift reduces performance and changes which model is most useful.**