___

Our account feature matrix has 26 features, so a binary feature-selection algorithm searches a much larger space than choosing weights for just three models.

Each candidate solution is a 26-bit mask:
- `1` means keep that feature.
- `0` means remove that feature.

There are \(2^{26}=67,108,864\) possible subsets, including the empty subset. We won't enumerate them all; the QIEA will search for promising candidates.

Our fitness function should balance predictive performance against model complexity. For example, we can use validation PR-AUC as the primary objective and penalize selecting too many features. Every promising subset must eventually be tested by retraining the model with those features.

___

### Step 1: Build the feature-selection evaluator

We'll use a two-stage approach:

1. Screening: run QIEA against a fast classical model using your existing 26 account features. This helps identify promising subsets without repeatedly running the much larger GNN.
2. Validation: retrain the GNN with the selected subset and compare it against the original 26-feature GNN.

The screening model is only a proxy. Its results must not be presented as the performance of the final GNN.

Create `src/models/qiea_feature_evaluator.py`

### Step 2 — Create the QIEA pilot runner

Create `src/models/run_qiea_pilot.py`.

This runner will use a smaller training sample to keep the first experiment manageable, while scoring each candidate on a fixed validation sample. It caches repeated subsets through your existing QIEA engine.

Output:

The QIEA pilot found a 12-feature subset that improves validation PR-AUC, so the feature-selection experiment is working

| Metric            | All 26 features | QIEA-selected |
| ----------------- | --------------- | ------------- |
| Validation PR-AUC | 0.074061        | 0.080913      |
| Validation F2     | 0.141743        | 0.133030      |
Interpretation: The selected subset improves PR-AUC, but its F2 score is lower. So it ranks positive cases better according to PR-AUC, but does not perform better on the F2 metric at the current classification threshold.

Because the evaluator uses logistic regression rather than our GNN ensemble, these numbers are a feature-screening result, not a direct comparison against the main detector.

___

We can make the stability experiment without changing the existing seed-42 results.

One detail matters: your script currently fixes both the data-sampling seed and the QIEA seed to `42`. For a proper stability test, we'll keep the training and validation samples fixed across runs while changing only the QIEA search seed. That way, differences are attributable to the search rather than different sampled data.

We'll also save each run in a separate folder, so the original pilot artifacts remain untouched.

Created `src/models/run_qiea_stability.py`

### Stability results

|Search seed|Features selected|Validation PR-AUC|Validation F2|
|---|---|---|---|
|42|12|0.080913|0.133030|
|11|13|0.083201|0.129095|
|22|13|0.081705|0.133270|
|33|10|0.084433|0.128498|

QIEA is finding subsets with consistently better PR-AUC than the original all-feature logistic-regression baseline, but the exact selected features vary between runs.

Across these four runs:

- Best PR-AUC: 0.084433, from seed 33.
- Lowest feature count: 10, from seed 33.
- Best F2: 0.133270, from seed 22.
- Each run evaluated around 68–73 unique subsets.

Seed 33 has the strongest PR-AUC, but we shouldn't automatically declare its 10-feature subset the winner. The scores are measured on the same validation sample that guides the search, so they may be optimistic.

These are promising candidates, particularly the outgoing amount features. However, repeated selection by this algorithm doesn't establish that a feature is genuinely important to detecting money mules.

The bigger finding is that QIEA's best PR-AUC varies from 0.0809 to 0.0844, while its feature subsets differ. That's enough to motivate a classical comparison.

___
### Now lets compare QIEA against a classical method

We'll use a simple, defensible baseline: train logistic regression on all 26 standardized features, rank features by the absolute magnitude of their learned coefficients, and evaluate the top 10, 12 and 13 features.

This gives us a conventional feature-selection baseline with the same training sample, validation sample and evaluation metrics as QIEA. Feature ranking will use the training data only.

Created `src/models/run_classical_feature_baseline.py`

|Method|Features|PR-AUC|F2|
|---|---|---|---|
|All-feature baseline|26|0.074061|0.141743|
|Classical coefficient ranking|10|0.079094|0.129150|
|Classical coefficient ranking|12|0.076621|0.131554|
|Classical coefficient ranking|13|0.075591|0.138273|
|QIEA, seed 42|12|0.080913|0.133030|
|QIEA, seed 33|10|0.084433|0.128498|

### What this tells us
- QIEA seed 33 achieved approximately 13.9% higher PR-AUC than the all-feature baseline.
- It also outperformed the classical top-10 subset's PR-AUC: `0.084433` versus `0.079094`.
- The classical top-13 subset had the highest F2 among the reduced-feature subsets, but the all-feature baseline still had the highest F2 overall.
- QIEA is optimizing PR-AUC, so its lower F2 is not necessarily a contradiction. These are different objectives.

However, the best QIEA score was selected from many candidate subsets using the same validation sample. That creates selection bias, so we cannot treat `0.084433` as an unbiased final performance estimate.

___

### What we'll do next

We'll run a controlled experiment to find out whether the 12 features selected by QIEA improve actual GNN performance—not just logistic regression performance.

The selected 12 features from your QIEA pilot are:

|Feature index|Feature name|
|---|---|
|0|`log1p_in_transaction_count`|
|2|`log1p_unique_senders`|
|7|`log1p_incoming_active_days`|
|9|`log1p_active_days`|
|13|`log1p_incoming_payment_format_diversity`|
|15|`log1p_receiving_currency_diversity`|
|16|`log1p_payment_currency_diversity`|
|20|`incoming_currency_abs_z_mean`|
|22|`outgoing_log_amount_mean`|
|23|`outgoing_log_amount_std`|
|24|`outgoing_log_amount_max`|
|25|`outgoing_currency_abs_z_max`|

These indices assume the column order in `feature_manifest.csv`, which must remain the source of truth.

## Experiment design

- Keep the original 26-feature checkpoints unchanged.
- Train the same GNN architectures using only the 12 selected columns.
- Keep the graph, train/validation/test masks, neighbor sampling, and training settings consistent.
- Save experimental checkpoints in a separate directory.
- Compare validation PR-AUC first, then evaluate the selected models on the test set.

One important detail: this is a feature-ablation experiment, not yet proof that QIEA is superior. We need to see whether the selected features help the GNN ensemble and compare that result against the existing 26-feature baseline.

Created `src/models/train_gnn_feature_subset.py`

Ran the code with the 3 models:
```
python .\src\models\train_gnn_feature_subset.py --model sage
python .\src\models\train_gnn_feature_subset.py --model gatv2
python .\src\models\train_gnn_feature_subset.py --model gin
```

### Results so far

QIEA-selected 12 features · test split · threshold 0.50

|Model|Val PR-AUC|Test PR-AUC|Test ROC-AUC|Test F2|
|---|---|---|---|---|
|GraphSAGE|0.3203|0.2779|0.9022|0.2193|
|GATv2|0.2702|0.2311|0.8803|0.1940|
|GIN|0.2117|0.1731|0.8678|0.1493|

GraphSAGE is the strongest of these three on test PR-AUC. That's a useful result, but we need to compare it against the same architectures trained with all 26 features before claiming feature selection helped.

### 2. What the numbers tell us

- GraphSAGE is the leading candidate. Its test PR-AUC is about 0.278, higher than the other two QIEA models.
- The models still flag many negatives at threshold 0.50. That is why precision is low despite recall being around 79–86%. We should choose an operating threshold using validation data rather than assume 0.50 is appropriate.
- Validation PR-AUC is higher than test PR-AUC for all three models. That gap is worth monitoring; it can reflect generalization differences and sampling variability.
- These results do not yet demonstrate a quantum advantage. QIEA is a quantum-inspired feature-selection method, and we still need the full-feature comparison.

Before building the new ensemble, let's establish whether the 12-feature models actually beat their full-feature counterparts.

Created `src/models/compare_feature_subset.py`

This compares the saved test metrics directly; it doesn't retrain anything

|Model|26-feature PR-AUC|QIEA-12 PR-AUC|Change|
|---|---|---|---|
|GraphSAGE|0.297555|0.277865|−6.61%|
|GATv2|0.240506|0.231083|−3.91%|
|GIN|0.204865|0.173133|−15.49%|

The percentage changes above are relative to each model's 26-feature PR-AUC.

### 2. What about F2?

|Model|26-feature F2|QIEA-12 F2|Result|
|---|---|---|---|
|GraphSAGE|0.218820|0.219329|Tiny improvement|
|GATv2|0.214940|0.194032|Worse|
|GIN|0.145462|0.149303|Tiny improvement|

The F2 changes are measured at the default threshold of 0.50. They don't establish that the 12-feature models are better overall. PR-AUC is more useful here because it evaluates ranking performance across thresholds, which matters with your imbalanced dataset.

The defensible conclusion is that QIEA reduced the feature count from 26 to 12, but the resulting GNNs had lower test PR-AUC in this experiment.

___

Now, We'll test whether the QIEA 12-feature predictions add useful information to the existing 26-feature ensemble, rather than retraining everything again.

Created `src/models/compare_qiea_ensembles.py`
It will search ensemble weights in increments of 0.05, select the best weights by validation PR-AUC, and choose the F2 threshold using validation predictions only. It will then evaluate those choices on the test set.

### Output

|Metric|Full 26 features|QIEA 12 features|Better|
|---|---|---|---|
|Test PR-AUC|0.3060|0.2773|Full|
|Test ROC-AUC|0.9147|0.9033|Full|
|Precision|0.1824|0.1673|Full|
|Recall|0.4535|0.4693|QIEA|
|F1|0.2602|0.2467|Full|
|F2|0.3496|0.3448|Full|
|Accounts flagged|1,579|1,781|—|
These results settle the first ensemble comparison: the original 26-feature ensemble remains the better choice overall. The QIEA 12-feature ensemble uses fewer features, but it hasn't matched the original model's ranking performance.

## What we've learned from Phase 9.3

- QIEA feature selection improved the pilot logistic regression's validation PR-AUC.
- That improvement did not transfer to the three GNNs on the test set.
- The full-feature ensemble achieved better test PR-AUC, ROC-AUC, precision, F1, and F2.
- The QIEA subset reduced input dimensionality by about 54%, from 26 features to 12, but that reduction came with a performance trade-off.

This is a legitimate experimental result. In your report, describe QIEA as a quantum-inspired feature-selection experiment, not as a demonstrated improvement over the classical baseline.

___

The **final** Phase 9.3 experiment: test whether the QIEA models add value when mixed with your original full-feature models.

Created `src/models/test_hybrid_qiea_ensemble.py`

### Final comparison

|Metric|Original 26-feature ensemble|Hybrid ensemble|Winner|
|---|---|---|---|
|Validation PR-AUC|0.355373|0.356874|Hybrid|
|Test PR-AUC|0.306027|0.305162|Original|
|Test ROC-AUC|0.914675|0.914812|Hybrid, marginally|
|Test precision|0.182394|0.217280|Hybrid|
|Test recall|0.453543|0.400000|Original|
|Test F1|0.260163|0.281596|Hybrid|
|Test F2|0.349599|0.342410|Original|
|Accounts flagged|1,579|1,169|Depends on objective|

The hybrid reduced flagged accounts by 410 and improved precision, but it missed more positive cases. Since F2 gives greater importance to recall, the original ensemble retains the advantage for your current evaluation objective.

___

### Final statement:

Completed three useful levels of evaluation:
1. QIEA feature selection against the classical logistic-regression baseline.
2. QIEA-selected features against the full-feature GNNs.
3. Hybrid ensemble testing to determine whether the reduced-feature models add complementary information.

The main finding is consistent: the 12-feature subset reduces dimensionality, but it hasn't improved overall test PR-AUC or F2 in your GNN experiments.

For your report, you can state that the quantum-inspired approach was implemented and evaluated, but no performance advantage was demonstrated. That's more defensible than claiming a quantum-inspired improvement based only on the pilot validation result.

One caveat: because the ensemble weights and thresholds were selected on reconstructed validation predictions, consider these results experimental. For a final rigorous report, avoid describing small differences as statistically significant without repeated-seed evaluation or uncertainty estimates.

