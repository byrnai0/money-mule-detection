___

Instead of asking:

> "What features does the model generally care about?"

we'll ask:

> **"Why did the model flag THIS particular account?"**

For example:

```
Account: 12345
Risk score: 0.94
Risk level: HIGH

Main contributing factors:
1. Very high outgoing maximum amount
2. High incoming payment-format diversity
3. High receiving-currency diversity
4. High outgoing transaction activity

Model evidence:
- Feature perturbation caused risk score to drop from 0.94 → 0.61
- Feature perturbation caused risk score to drop from 0.94 → 0.73
...
```

___

The process goes like:

```
account ID
    ↓
load its 26 features
    ↓
run SAGE + GATv2 + GIN
    ↓
weighted ensemble (0.78 / 0.19 / 0.03)
    ↓
risk score
    ↓
perturb each feature individually
    ↓
measure score change
    ↓
rank features
    ↓
generate explanation
```

___

Created `individual_explanation.py`
The script will:

1. Load `training_graph.pt`
2. Load the three sampled models
3. Take an account ID
4. Calculate its ensemble risk score
5. Perturb each of the 26 features
6. Measure how much the risk score changes
7. Rank the features by impact
8. Convert the top features into human-readable reasons
9. Save the explanation to CSV

run `python src\models\individual_explanation.py --account 100`


### Output:

```
============================================================
ACCOUNT EXPLANATION
============================================================
Account index : 100
Account ID    : 10_800059840
Risk score    : 0.510683
Risk level    : MEDIUM

Main factors:
1. Receiving currency diversity (increased risk, score change +0.4304)
2. Incoming payment format diversity (reduced risk, score change -0.2594)
3. Incoming log amount mean (increased risk, score change +0.2201)
4. Out transaction count (reduced risk, score change -0.1774)
5. Unique receiver banks (reduced risk, score change -0.1683)
```

___

One thing I want to fix before we call 8B fully done: **the wording "increased risk" / "reduced risk" is backwards for negative perturbation changes in some cases.** The score change is:

> baseline score − perturbed score

So `+0.4304` means removing that feature caused the score to fall by 0.4304, which indicates the feature was **supporting the higher risk score**. That's fine. But a negative value means removing the feature made the score _increase_, meaning the original feature was suppressing the risk score. We should make that explicit in the final explanation.


Result:

```
Account ID: 10_800059840
Risk score: 0.510683
Risk level: MEDIUM
```

|Rank|Feature|Score change|
|---|---|--:|
|1|Receiving currency diversity|+0.4304|
|2|Incoming payment format diversity|−0.2594|
|3|Incoming log amount mean|+0.2201|
|4|Out transaction count|−0.1774|
|5|Unique receiver banks|−0.1683|
### What this actually tells us

For this account, **receiving currency diversity is the strongest positive contributor** according to this perturbation test. Removing it caused the ensemble risk score to drop substantially, from `0.5107` to roughly `0.0803`.

The negative contributions mean the opposite: removing those features caused the predicted risk to **increase**, so those features were suppressing the model's risk score for this particular account.

This is **model attribution**, not proof that any of these behaviors are themselves suspicious.

(a graph is saved at`data/graphs/evaluation/individual_explanations.csv`)

