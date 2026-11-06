___

First read the [[Phase 9.0.0 - TLDR first]]
### So which one should we choose?

Structure for Phase 9:

```
9.1  Define binary optimization problem 
 │
 ▼
9.2  Implement Q-bit / QIEA representation
 │
 ▼
9.3  Implement candidate generation
 │
 ▼
9.4  Implement fitness evaluation
 │
 ▼
9.5  Run quantum-inspired search
 │
 ▼
9.6  Select optimized feature subset
 │
 ▼
9.7  Retrain GNN models
 │
 ▼
9.8  Compare against 26-feature baseline
 │
 ▼
9.9  Analyze optimization results
 │
 ▼
9.10  Quantum-inspired ensemble-weight optimization
 │
 ▼
9.11  CP-decomposition graph pooling
```

Why this method?
Because we can make a very clear experiment:

### Baseline

```
26 features
+
GraphSAGE/GATv2/GIN
+
classical ensemble
```

versus:

### Quantum-inspired version

```
optimized feature subset
+
same GNN architecture
+
same training/evaluation procedure
```

Then we measure:

```
PR-AUC
ROC-AUC
Precision
Recall
F1
F2
```

If the quantum-inspired subset gets better F2/PR-AUC with fewer features, **that's a meaningful result**.

If it doesn't beat the baseline, that's also a valid research result.

___

### What algorithm should we use?

A reasonable candidate is a **Quantum-Inspired Evolutionary Algorithm (QIEA/QEA)**.

Conceptually:

```
Initialize probability/Q-bit representation
              ↓
Generate binary candidate solutions
              ↓
Evaluate candidates
              ↓
Keep good solutions
              ↓
Update probability amplitudes
              ↓
Generate new candidates
              ↓
Repeat
              ↓
Best binary solution
```

The "quantum" inspiration comes from representing a binary solution probabilistically rather than simply maintaining one fixed binary chromosome.

For example, conceptually:

```
Q-bit:
|α|² = probability of 0
|β|² = probability of 1
```

with:

$$
|\alpha|^2+|\beta|^2=1
$$
Then a measurement produces 0 or 1.

___

What we're going to build:

Your current Phase 6 already searches 5,151 possible ensemble weight combinations in increments of `0.01` and chooses the combination with the highest validation PR-AUC. That is a solid baseline, but it is an exhaustive grid search—not a quantum-inspired optimizer.

For Phase 9, we'll build a QUBO-based ensemble weight optimizer and compare it against your existing tuned voting method.

| Component              | Existing Phase 6              | Proposed Phase 9                                |
| ---------------------- | ----------------------------- | ----------------------------------------------- |
| Base models            | GraphSAGE, GATv2, GIN         | Same three models                               |
| Weight selection       | Exhaustive grid search        | QUBO + classical simulated annealing            |
| Optimization objective | Validation PR-AUC             | Quadratic validation prediction-error objective |
| Final evaluation       | PR-AUC, ROC-AUC, F1, F2, etc. | Same metrics, on the same held-out test split   |

___

## 1. How the QUBO will work

We'll represent each ensemble weight in increments of `0.01`. For example:

- GraphSAGE: 78 units → weight `0.78`
- GATv2: 19 units → weight `0.19`
- GIN: 3 units → weight `0.03`

The total must equal 100 units, so the weights sum to 1.

The optimizer will minimize the validation Brier score, which measures the squared difference between predicted probabilities and actual labels. This gives us a quadratic objective suitable for QUBO.

Conceptually, our objective is:
$$
\begin{gathered}
\min_{\mathbf{w}} \frac{1}{N} \sum_{n=1}^{N} \left( \sum_{i=1}^{3} w_i p_{ni} - y_n \right)^2
\end{gathered}
$$
subject to:
$$
w_i \geq 0, \quad \sum_{i=1}^{3} w_i = 1
$$
The weight constraint will be incorporated into the binary optimization formulation.

Why not directly optimize PR-AUC? PR-AUC depends on the ranking of predictions and is not naturally a quadratic objective. We'll optimize a quadratic surrogate, then evaluate the resulting ensemble using PR-AUC, ROC-AUC, precision, recall, F1 and F2. We'll explicitly report this distinction in the project documentation.

## 2. Keep the existing pipeline unchanged

Our current files already give us what we need:

- `ensemble_predictions.csv` — aligned validation and test probabilities for all three GNNs.
- `tuned_voting.py` — the existing PR-AUC grid-search baseline.
- `training_graph.pt` — labels and graph data, although the QUBO optimizer can obtain the labels directly from the prediction CSV.

We'll add a separate file:

`src/models/qubo_ensemble_optimizer.py`

It will:

1. Load the existing prediction CSV.
2. Construct the quadratic objective using validation predictions only.
3. Run a classical simulated-annealing search on the QUBO.
4. Verify that the resulting weights are valid and sum to 1.
5. Compare the optimized weights with your existing `0.78 / 0.19 / 0.03` baseline.
6. Evaluate the frozen weights on the test set only after optimization.
7. Save the weights, metrics and comparison report.

One methodological point matters here: your current tuned voting already tests 5,151 combinations. We'll compare the QUBO solution against that result, not just claim success because the optimizer found some weights. Since the objectives differ, we should report both the Brier score and the AML ranking metrics.

___

### Phase 9 structure

[[Phase 9.0.0 - TLDR first]]
[[Phase 9.1 - Define the optimization problem]]
[[Phase 9.2 - QIEA configuration]]
[[Phase 9.3 - Quantum-inspired feature selection]]
[[Phase 9.4 - CP-decomposition graph pooling]]
[[Phase 9.5 - Consolidation]]
[[Phase 9 Report by GPT]]
[[Requirements Before Phase 9]]
