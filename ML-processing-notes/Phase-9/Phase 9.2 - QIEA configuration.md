___

One important correction before we code: we should not assume that masking features in a model trained on all 26 features accurately predicts how a model retrained on a smaller subset will perform. We'll use masking to shortlist candidates, then retrain and validate the selected subset before making any performance claims.

| Setting                   | Initial value       | Why                                    |
| ------------------------- | ------------------- | -------------------------------------- |
| Number of features        | 26                  | Existing feature matrix                |
| Population size           | 12                  | Keeps the initial search manageable    |
| Maximum generations       | 15                  | 180 candidate evaluations at most      |
| Minimum selected features | 5                   | Avoids tiny, uninformative subsets     |
| Maximum selected features | 22                  | Encourages feature reduction           |
| Random seed               | 42                  | Reproducible experiments               |
| Fitness                   | Validation PR-AUC   | Appropriate for severe class imbalance |
| Secondary metric          | Validation F2       | Tracks recall-oriented performance     |
| Elitism                   | Keep best candidate | Prevents losing the best solution      |
___

Create the optimizer engine: `qiea_feature_selector.py`

This file will handle the search algorithm independently of the GNN models. That means we can test the optimizer before connecting it to your real validation pipeline.

What this code does:

- Q-bit-inspired representation: Each feature has an angle \(\theta\). Its selection probability is \(\sin^2(\theta)\).
- Sampling: Generates binary feature subsets from those probabilities.
- Repair: Enforces the 5–22 feature constraint.
- Fitness: Uses PR-AUC to rank candidates. F2 is recorded as a secondary metric.
- Elitism: Preserves the best subset found so far.
- Rotation update: Adjusts the angles toward the best-known binary solution.
- Caching: Avoids evaluating the same feature subset twice.

One limitation to be aware of: this is a simplified Q-bit-inspired evolutionary optimizer, not a complete reproduction of a specific published QIEA. We should document that distinction in the report and compare it against a classical baseline.

___

Before testing the actual values, I decided to smoke test the search engine using a small synthetic evaluator. This checks that the code runs, obeys the feature-count limits, and improves a known toy objective. It does not test money-mule detection performance.

Created: `src/models/test_qiea_engine.py`

Output:

```
Selected indices: [0 2 4 6]
Selected count: 4
Best metrics: {'pr_auc': 1.0, 'f2': 1.0}
Unique subsets evaluated: 44
```

___

## One important decision: what should the QUBO optimize?

I'd now prioritize a QUBO-based ensemble optimizer as the main quantum-inspired component, consistent with the project handoff's literature-based plan.

The distinction is:

- Existing system: combines three trained GNNs using fixed soft-voting weights.
    
- Proposed system: formulates ensemble selection or weighting as a binary optimization problem, then solves it with a classical optimizer inspired by quantum optimization.
    
- Evaluation: compares the optimized ensemble against the existing tuned-voting baseline on the same validation split, with the test split held back for final evaluation.
    

### How we'll formulate it

There are only three base models, so selecting a subset alone would give us just \(2^3=8\) combinations. Brute-forcing eight combinations would be trivial and would not justify a sophisticated optimizer.

A more meaningful experiment is to optimize discretized ensemble weights.

For example:

|Model|Existing weight|Optimized weight|
|---|---|---|
|GraphSAGE|0.78|To be determined|
|GATv2|0.19|To be determined|
|GIN|0.03|To be determined|

We can represent candidate weights using binary variables and construct a QUBO objective using validation prediction error, with a penalty enforcing the weight constraint. Then we'll evaluate the resulting ensemble using PR-AUC, F2, precision and recall.

Important: the QUBO's optimization objective and our final AML evaluation metrics are not necessarily the same. We'll report that distinction clearly rather than claiming that optimizing one automatically maximizes the other.

___
## Step 1 — Define the optimization properly

We'll use your three existing GNN probability columns from `data/graphs/ensemble_predictions.csv`.

The optimizer will choose weights \(w_s,w_a,w_g\) for GraphSAGE, GATv2 and GIN, respectively. Each weight will be a multiple of `0.01`, and the weights must sum to 1.

Our quadratic objective will be the validation Brier score:

$$
L(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} (w_s p_{ns} + w_a p_{na} + w_g p_{ng} - y_n)^2
$$
We'll encode the weights as binary variables and add a penalty for violating the sum-to-100 constraint.

Important: this optimizes squared probability error, not PR-AUC directly. We'll compare both metrics, plus your existing classification metrics, to determine whether the QUBO experiment actually helps.

Created: `src/models/qubo_ensemble_optimizer.py`

Results comparison:

| Metric                       | Existing weights   | QUBO weights       | Better result |
| ---------------------------- | ------------------ | ------------------ | ------------- |
| Weights (SAGE / GATv2 / GIN) | 0.78 / 0.19 / 0.03 | 0.68 / 0.32 / 0.00 | —             |
| Brier score ↓                | 0.114922           | 0.114019           | QUBO          |
| PR-AUC ↑                     | 0.308526           | 0.307865           | Existing      |
| ROC-AUC ↑                    | 0.915761           | 0.914909           | Existing      |
| Precision ↑                  | 0.058385           | 0.059634           | QUBO          |
| Recall ↑                     | 0.839370           | 0.831496           | Existing      |
| F1 ↑                         | 0.109177           | 0.111287           | QUBO          |
| F2 ↑                         | 0.228383           | 0.231701           | QUBO          |
| Accounts flagged             | 9,129              | 8,854              | —             |
The Brier score measures probability prediction error, so lower is better. PR-AUC measures ranking quality, which is especially relevant when illicit accounts are rare.

___

Created `compare_qubo_thresholds.py`

Run it. 
These results give us a clearer picture. The QUBO ensemble improves precision, but the existing ensemble still wins on test F2, recall, PR-AUC and ROC-AUC. So we have a trade-off, not an overall winner.

### Fair comparison after threshold tuning

Both thresholds were selected using validation F2 and then frozen for the test evaluation.

|Test metric|Existing ensemble|QUBO ensemble|Winner|
|---|---|---|---|
|Threshold|0.818|0.826|—|
|Precision|18.73%|19.98%|QUBO|
|Recall|44.41%|41.57%|Existing|
|F1|0.2634|0.2699|QUBO|
|F2|0.3485|0.3419|Existing|
|PR-AUC|0.3085|0.3079|Existing|
|ROC-AUC|0.9158|0.9149|Existing|
|Accounts flagged|1,506|1,321|—|

At these thresholds, QUBO flags 185 fewer accounts and has higher precision. However, it also misses more illicit accounts, resulting in lower recall and F2.

The validation F2 scores were very close: `0.369876` for the existing ensemble and `0.371212` for QUBO. That small validation advantage did not carry over to test F2.

Before adding more complexity, let's test whether the QUBO result is stable across random seeds and constraint-penalty settings. One annealing run isn't enough to assess how reliably the optimizer finds good weights.
We'll run a small robustness experiment using validation predictions only:

- 5 constraint penalties: `0.001`, `0.01`, `0.1`, `1.0`
    - That's actually four settings; we'll test those four initially.
- 5 random seeds per setting.
- Record the best feasible weights, validation Brier score, PR-AUC and F2.
- Check whether the optimizer repeatedly chooses similar weights or produces very different solutions.

This will help us determine whether the `0.68 / 0.32 / 0.00` result is reproducible or just one outcome of simulated annealing.

Create `src/models/qubo_robustness.py`
This runs 20 annealing experiments, each using 500 reads and 1,000 sweeps. It should be less expensive than repeating your original full configuration for every seed.

### What these results tell us

Three conclusions:

- The constraint penalty matters. At `0.001`, the weights vary wildly and average performance is worse. At higher penalties, the solutions are much more consistent.
- GIN receives zero weight in all 15 runs with penalties of `0.01` or higher. This suggests the Brier-score objective favors a two-model ensemble for this validation set.
- Stability is not the same as superiority. Our previous test results still showed the existing ensemble winning on PR-AUC and F2. The robustness experiment doesn't overturn that.

___

Create `compare_brier_grid.py`


We'll compare:

- QUBO: simulated annealing minimizing validation Brier score.
- Classical grid search: exhaustive search over all 5,151 weight combinations, also minimizing validation Brier score.
- Existing baseline: your original `0.78 / 0.19 / 0.03` weights.

All weight configurations will be selected using validation data only. Test data will be used for evaluation after the weights are frozen.

### What we're looking for

- If the classical search finds a better Brier score than QUBO, that quantifies the gap between the heuristic and the best solution on the same discrete grid.
- If QUBO finds the same weights or nearly identical performance, it demonstrates that the heuristic can recover a strong solution, but not that it beats exhaustive search.
- If both perform similarly to your original tuned ensemble, then the main finding is that optimizing Brier score produces a different trade-off from optimizing PR-AUC.

Run it.

Output:

Now we have the comparison we needed. The classical exhaustive search finds a slightly better Brier-score solution than QUBO, while the original PR-AUC-tuned ensemble still has the best test PR-AUC.

This is a valuable result for your project because we can now distinguish what the quantum-inspired optimizer achieves from what a simpler classical method achieves.

### Final test-set comparison

|Metric|Classical Brier grid|QUBO|Existing PR-AUC-tuned|
|---|---|---|---|
|Weights (SAGE / GATv2 / GIN)|0.66 / 0.34 / 0.00|0.68 / 0.32 / 0.00|0.78 / 0.19 / 0.03|
|Brier score ↓|0.114019|0.114019*|0.114922|
|PR-AUC ↑|0.307499|0.307865|0.308526|
|ROC-AUC ↑|0.914709|0.914909|0.915761|
|Precision ↑|0.059878|0.059634|0.058385|
|Recall ↑|0.831496|0.831496|0.839370|
|F1 ↑|0.111711|0.111287|0.109177|
|F2 ↑|0.232435|0.231701|0.228383|

The classical search checks all 5,151 valid weight combinations. Its selected weights, `0.66 / 0.34 / 0.00`, are very close to the QUBO weights of `0.68 / 0.32 / 0.00`.

That means simulated annealing found a strong solution, but this experiment does not show it outperforming exhaustive search. Also, the classical Brier-optimal weights are not the same as the PR-AUC-optimal weights—because those are different objectives.

### What we've established

- QUBO implementation works: it produces feasible weights and repeatable solutions with suitable constraint penalties.
- Classical optimization is at least as strong here: exhaustive search finds a slightly better Brier-score solution for the same discrete weight space.
- The objective matters: the existing ensemble has better test PR-AUC and ROC-AUC, while the Brier-oriented ensembles achieve slightly better precision and F2 at threshold `0.50`.
- No quantum advantage has been demonstrated: this is a classical quantum-inspired heuristic, and the search space is small enough for exhaustive search.

___

Ran another testing sequence, the latest results tell us
The threshold-tuned comparison is now complete. The classical Brier-grid ensemble does not beat the existing ensemble on test F2, even though it had the highest validation F2 of the three.

|Test metric|Existing ensemble|QUBO|Classical Brier grid|
|---|---|---|---|
|PR-AUC ↑|0.308526|0.307865|0.307499|
|ROC-AUC ↑|0.915761|0.914909|0.914709|
|Precision ↑|18.73%|19.98%|18.45%|
|Recall ↑|44.41%|41.57%|44.57%|
|F1 ↑|0.2634|0.2699|0.2609|
|F2 ↑|0.3485|0.3419|0.3473|
|Flagged accounts|1,506|1,321|1,534|

The existing PR-AUC-tuned ensemble remains the strongest overall choice if your main priorities are ranking suspicious accounts and maintaining recall.

The QUBO ensemble gives higher precision and F1, but at the cost of missing more illicit accounts. The classical Brier-grid ensemble gets slightly higher recall than the existing ensemble, but its F2 is lower.

**Phase 9B conclusion**: the QUBO pipeline works, but it has not demonstrated a consistent advantage over the existing classical ensemble. We should document that honestly rather than keep tuning until we find a favorable result.