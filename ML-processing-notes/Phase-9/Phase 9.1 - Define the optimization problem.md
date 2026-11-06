___

We have **26 features**.

For every feature, create a binary decision:

```
xᵢ = 1 → use feature
xᵢ = 0 → don't use feature
```

So one possible solution could look like:

```
[1, 1, 0, 1, 0, 0, 1, ...]
```

For example:

```
Feature                          Selected?
------------------------------------------------
outgoing_log_amount_max              1
incoming_log_amount_mean             1
activity_burst_ratio                 0
unique_receiver_banks                1
currency_diversity                   0
...
```

With 26 features, our optimizer is searching through:

$$2^{26} = 67,108,864$$

possible feature subsets.

That's the actual optimization problem.

And this is where the quantum-inspired algorithm becomes useful.

___

### What exactly is "quantum-inspired" here?

The algorithm borrows concepts from quantum computation, particularly the idea of a **Q-bit probability representation**.

Instead of initially saying:

```
feature 1 = definitely selected
feature 2 = definitely removed
feature 3 = definitely selected
...
```

we maintain probabilities.

For example:

```
Feature                Probability selected
------------------------------------------------
F1                         0.50
F2                         0.50
F3                         0.50
F4                         0.50
...
```

Then the optimizer samples possible solutions.

Example:

```
Probability vector
       ↓
[0.5, 0.5, 0.5, 0.5]
       ↓
sample
       ↓
[1, 0, 1, 1]
```

Evaluate it.

If that solution performs well, the algorithm increases the probability of making similar choices.

Eventually:

```
F1 → 0.94
F2 → 0.08
F3 → 0.91
F4 → 0.87
```

Meaning:

```
F1 → probably keep
F2 → probably remove
F3 → probably keep
F4 → probably keep
```

That's the **quantum-inspired part**.

The probabilities aren't quantum probabilities physically existing in a quantum computer. They're classical numbers representing the Q-bit-inspired search state.

Visual of the algo:

```
                 INITIALIZE
                     │
                     ▼
          Q-bit probability vector
          [0.5 0.5 0.5 ... 0.5]
                     │
                     ▼
             Generate candidates
                     │
                     ▼
             Evaluate candidates
                     │
                     ▼
              Find best solution
                     │
                     ▼
          Update Q-bit probabilities
                     │
                     ▼
             Generate new candidates
                     │
                     ▼
                  Repeat
                     │
                     ▼
              BEST FEATURE SET
```

___

### Main optimization metric is PR-AUC

Why?

Because it focuses on the precision/recall tradeoff and is much more informative for highly imbalanced fraud/AML detection.

Our current baseline is:

```
PR-AUC = 0.308526
```

Then we'll use **F2** as an important secondary metric because we care more about catching suspicious accounts than maximizing precision.

Current baseline:

```
F2 = 0.342324
Recall = 0.467717
Precision = 0.165184
```

So conceptually:

```
Primary objective:
        ↑ PR-AUC

Secondary:
        ↑ F2
        ↑ Recall

Constraint:
        Don't select an absurd number of features
```

Due to computational problems of the GPUs, we're going to use a **two-stage optimization strategy**

#### Stage 1 — Cheap search

(This is technically a **proxy fitness evaluation**, not the final feature-selection experiment. Gotta say sum like "A frozen-model surrogate evaluation was used during the quantum-inspired search to reduce computational cost. The selected feature subset was subsequently retrained and independently evaluated to verify the optimization result.")

Use the already-trained models and mask features.

For a candidate:

```
[1, 1, 0, 1, 0, ...]
```

we set the unused standardized features to zero:

```
selected feature → original value
removed feature  → 0
```

Then run inference and calculate validation performance.

This allows QIEA to cheaply explore many candidate subsets.

#### Stage 2 — Actual confirmation

Once QIEA finds the best subset:

```
QI optimizer
      ↓
Best feature subset
      ↓
Example:
17 / 26 features
```

we **actually retrain the GNNs using only those features**.

Then we compare:

```
                    Baseline       QI optimized
---------------------------------------------------
Features              26              17
PR-AUC              0.3085          ?
ROC-AUC              0.9158          ?
Precision            0.1652          ?
Recall               0.4677          ?
F1                   0.2441          ?
F2                   0.3423          ?
```

That second result is the one that matters.

If the optimized model genuinely performs better after retraining, **then we have evidence that the quantum-inspired optimization actually helped**.

___

### The Q-bit representation

For each feature we maintain something like:
$$
|\psi_i\rangle = \alpha_i|0\rangle + \beta_i|1\rangle
$$
where:
$$
|\alpha_i|^2 + |\beta_i|^2 = 1
$$
In implementation, we're essentially interested in:

```
P(feature = 0)
P(feature = 1)
```

For example:

```
Feature                    P(0)       P(1)
------------------------------------------------
F1                         0.20       0.80
F2                         0.85       0.15
F3                         0.45       0.55
F4                         0.10       0.90
```

The algorithm samples feature subsets according to these probabilities.

### How the evolution works

Suppose the optimizer discovers:

```
Candidate A

[1, 1, 0, 1, 0, 1]
```

and this candidate has excellent PR-AUC.

The Q-bit probabilities are then shifted toward:

```
[1, 1, 0, 1, 0, 1]
```

So the next generation is more likely to produce similar solutions.

Over multiple generations:

```
Generation 1
0.50 0.50 0.50 0.50 0.50 ...

        ↓

Generation 10
0.72 0.63 0.28 0.81 0.31 ...

        ↓

Generation 20
0.91 0.84 0.09 0.94 0.14 ...

        ↓

Final
1    1    0    1    0 ...
```

This gives us an adaptive search rather than blindly checking combinations.

TLDR: "We formulate feature selection as a binary optimization problem and employ a quantum-inspired evolutionary algorithm to search for a high-performing subset of account-level graph features. The algorithm runs classically but uses Q-bit-inspired probabilistic representation and evolutionary updates to guide the search."

___

# Note:

**QUBO and QIEA are not the same thing.**

QUBO is a **mathematical formulation**:

```
binary variables + linear terms + quadratic interactions
```

QIEA is an **optimization algorithm**:

```
probabilistic Q-bit population
        ↓
candidate solutions
        ↓
fitness
        ↓
update
        ↓
repeat
```

We don't need to force a QUBO into the project just because the word "quantum" appears in the title. For our actual feature-selection objective, **QIEA lets us optimize the validation metric directly**, which is much cleaner.

