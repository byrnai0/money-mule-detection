___

Instead of asking:

> "What do false positives look like on average?"

we want:

> **"Which individual accounts are the model most confidently getting wrong?"**

We're going to extract:

```
Highest-risk false positives
Lowest-risk false negatives
Highest-risk true positives
Lowest-risk true positives
```

This is particularly useful for a graph-based AML project because those cases can reveal what the model is actually learning.

For example:

```
FP with score 0.999
```

is much more interesting than a generic FP with score 0.80.

Likewise:

```
FN with score 0.05
```

tells us which illicit accounts look extremely legitimate to the current system.

___

Created `deep_error_analysis.py`. Run it.

Output is a lot. Its saved as a csv file. Check it.

___

Cases that should be as the output:

### High-confidence false positives

Something like:

```
account X
risk = 0.9998
actual = legitimate
```

If these repeatedly have:

- huge incoming activity
- many senders
- lots of banks
- high currency diversity
- bursty behavior

then that's evidence that **legitimate high-volume financial hubs resemble laundering networks** in this dataset.

### Low-confidence false negatives

Something like:

```
account Y
risk = 0.07
actual = illicit
```

Those are the dangerous misses.

If they consistently have:

- few counterparties
- low volume
- little bank diversity
- no reciprocal relationships

then we have identified a genuine blind spot:

> **the current graph/features are much better at finding network-visible illicit accounts than quiet/small illicit accounts.**

___

(GPT SHIT)
## What the deep error analysis actually tells us

This is more interesting than the earlier group-average analysis.

### The most confident false positives

The top 25 false positives have an average model score of:

```
0.9857
```

That is **extremely high**. These aren't borderline mistakes; the model is very confident that these legitimate accounts are illicit.

And their behavior explains why.

Compared with the legitimate population, they have:

```
incoming activity      → high
unique senders         → high
burst ratio            → high
amount statistics      → high
counterparty behavior  → suspicious-looking
```

For example, their average `log1p_unique_senders` is about **1.45**, and their average burst ratio is about **1.79**. Their incoming/outgoing amount maxima are also very large. So these are legitimate accounts whose transaction behavior genuinely resembles the learned illicit patterns.

That gives us a useful interpretation:

> **The model's false positives are not random; many are high-activity or highly connected legitimate accounts that resemble laundering behavior structurally and transactionally.**

### The lowest-risk false negatives

This is the scarier group.

The 25 lowest-scoring false negatives have an average risk score of only:

```
0.1731
```

with some as low as:

```
0.0563
```

So these illicit accounts look **very legitimate to the current model**.

Their biggest distinguishing characteristic is much weaker network breadth:

```
log1p_unique_senders ≈ 0.83
log1p_unique_receivers ≈ 0.90
reciprocity ≈ 0
```

They also have much higher concentration:

```
FN concentration ≈ 0.611
```

compared with the high-confidence TP group at around:

```
0.286
```

So we're seeing a fairly clear pattern:

```
Easy-to-detect illicit account
→ more distributed / network-visible activity

Hard-to-detect illicit account
→ fewer counterparties
→ little/no reciprocal activity
→ more concentrated relationships
```

That's a **real blind spot** in our current feature/graph representation.

It doesn't mean the model is bad. It tells us exactly what kind of laundering behavior it currently struggles with.

---

# Something else interesting: high-risk TPs vs low-risk TPs

Even among accounts the model gets correct, there's a huge difference.

### Highest-risk true positives

Average score:

```
0.9998
```

These are basically screaming "suspicious" according to the model.

### Lowest-risk true positives

Average score:

```
0.8122
```

These are still correctly above our threshold, but only barely.

Their transaction volume is much higher than the high-risk TP group:

```
log1p_in_transaction_count
low-risk TP:   2.66
high-risk TP:  1.60
```

That is interesting because it suggests **more activity alone isn't necessarily what drives the highest confidence**.

The model seems to care about the **combination of behavioral/network patterns**, not merely transaction count.