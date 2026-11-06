___

**We are not putting the 515k-node graph into a quantum computer. We are using quantum-inspired optimization to search for a better configuration of our fraud-detection system.**

**"We're formulating feature selection as a binary optimization problem and using a quantum-inspired evolutionary optimization algorithm, executed classically, to search for an optimal feature subset."**

Current setup is:

```
Transactions
     ↓
26 engineered features
     ↓
Graph
     ↓
GNN models
 ┌─────────┬─────────┬─────────┐
 │ GraphSAGE│ GATv2  │   GIN   │
 └─────────┴─────────┴─────────┘
     ↓
Ensemble
     ↓
0.78 SAGE
0.19 GATv2
0.03 GIN
     ↓
Risk score
     ↓
Threshold 0.79
     ↓
Illicit / Legitimate
```

After Phase 9, the full setup is:

```
                 CURRENT SYSTEM
                      │
                      ▼
              26 engineered features
                      │
                      ▼
              ┌───────────────┐
              │ GNN Ensemble  │
              │ SAGE          │
              │ GATv2         │
              │ GIN           │
              └───────┬───────┘
                      │
                      ▼
                 Validation
                   F2/PR-AUC
                      │
                      ▼
             ┌─────────────────┐
             │ QUBO formulation│
             └────────┬────────┘
                      │
                      ▼
          Binary feature-selection
                 variables
                      │
                      ▼
          Quantum-Inspired Optimizer
                      │
             ┌────────┴────────┐
             │                 │
        candidate 1        candidate 2
             │                 │
             └────────┬────────┘
                      ▼
             Evaluate candidates
                      │
                      ▼
               Update search
                      │
                     ...
                      │
                      ▼
              BEST FEATURE SUBSET
                      │
                      ▼
             Retrain same GNNs
                      │
                      ▼
                Test evaluation
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Classical              Quantum-inspired
       baseline                  model
          │                       │
          └───────────┬───────────┘
                      ▼
                  Compare
```

___
### So what is being done here?

Imagine we have three models.

We could represent whether a model is selected using binary variables:

```
x₁ = GraphSAGE selected
x₂ = GATv2 selected
x₃ = GIN selected
```

Each can only be:

```
0 = don't select
1 = select
```

That's exactly the kind of binary decision structure that QUBO is designed around. A QUBO uses binary variables and an objective containing linear and pairwise terms.

For example:

```
x₁ = 1
x₂ = 1
x₃ = 0
```

means:

> use GraphSAGE + GATv2, don't use GIN.

But there's an immediate problem.

Our actual ensemble doesn't just select models. It assigns **weights**:

```
SAGE  = 0.78
GATv2 = 0.19
GIN   = 0.03
```

So we need to decide **what exactly we're going to optimize**.

___

Now instead of optimizing all 26 features + model weights + threshold simultaneously, first we finna optimize the ensemble config.

Something like:

```
GraphSAGE
GATv2
GIN
```

with discretized weight choices.

For example, weights could come from:

```
0.00
0.05
0.10
...
1.00
```

subject to:

```
wSAGE + wGATv2 + wGIN = 1
```

Our classical solution is currently:

```
(0.78, 0.19, 0.03)
```

We'll ask the quantum-inspired optimizer:

> Find the combination of ensemble weights that minimizes our objective.

___
### So whats the objective then?

Our dataset is heavily imbalanced.

From my evaluation:

```
Illicit prevalence ≈ 1.23%
```

So optimizing ordinary accuracy would be useless.

A model predicting:

```
everyone = legitimate
```

would get roughly 98.8% accuracy.

That's why we've already been using:

- PR-AUC
- F1
- F2
- Recall
- Precision

For AML/money-mule detection, we care significantly about **catching illicit accounts**, so F2 is particularly useful because it weights recall more heavily than precision.

Therefore our optimization objective can be based around `maximize F2`
or equivalently `minimize -F2`

___

### QUBO?

A QUBO looks conceptually like:
$$
E(x) = \sum_{i} a_i x_i + \sum_{i<j} b_{ij} x_i x_j
$$
where 
xᵢ ∈ {0,1}

The optimizer searches for the binary configuration with the lowest energy.

[D-Wave's documentation](https://docs.dwavequantum.com/en/latest/shared/parameters.html) describes exactly this structure: diagonal coefficients represent individual-variable biases, while off-diagonal coefficients represent interactions between pairs of variables.

The [Pasqal QUBO](https://docs.pasqal.com/applicationsolvingtools/qubo/) docs is also particularly good for understanding this without drowning in quantum physics.

___

### So how do we turn our problem into binary variables?

Suppose we discretize each model's weight.

Instead of:

```
wSAGE = 0.78
```

we represent possible weight levels with binary variables:

```
SAGE_0
SAGE_5
SAGE_10
...
SAGE_100
```

where:

```
SAGE_0 = 1
```

means:

> GraphSAGE gets weight 0%.

and:

```
SAGE_80 = 1
```

means:

> GraphSAGE gets weight 80%.

We then enforce that exactly **one weight level** is selected for each model.

Same for GATv2 and GIN.

So the quantum-inspired optimizer isn't magically understanding:

> "GraphSAGE is better."

We're **encoding our problem into binary decisions** and asking an optimizer to search those decisions efficiently.

That's the key concept.

___

### But there's another approach

We could instead optimize **feature selection**.

We have 26 features:

```
f1
f2
f3
...
f26
```

Define:

```
xᵢ = 1 → feature selected
xᵢ = 0 → feature removed
```

Then the optimizer searches:

```
[1,1,0,1,0,1,...]
```

for the subset that gives the best validation performance.

This is actually a **very natural QUBO problem** because feature selection is a binary combinatorial optimization problem. QUBO formulations are commonly used for feature selection and other combinatorial problems. [MathWorks]([https://www.mathworks.com/help/matlab/qubo.html?utm_source=chatgpt.com](https://www.mathworks.com/help/matlab/qubo.html))

And my Phase 8.1 results give us a very useful starting point:

```
outgoing_log_amount_max
incoming_payment_format_diversity
receiving_currency_diversity
outgoing_log_amount_std
out_transaction_count
...
```

We already know which features appear important.



# The resources I want you to actually read

Don't read 30 papers. Read these in roughly this order.

### 1. **QUBO — easiest starting point**

[Pasqal — A Tour of QUBO](https://docs.pasqal.com/applicationsolvingtools/qubo/tutorial/00-a-tour-of-qubo/?utm_source=chatgpt.com)

This is probably the **best first read for you**. It explains QUBO as a binary optimization problem without immediately drowning you in quantum mechanics. [Pasqal Documentation](https://docs.pasqal.com/applicationsolvingtools/qubo/tutorial/00-a-tour-of-qubo/?utm_source=chatgpt.com)

### 2. **QUBO formulation — deeper/academic**

[A Tutorial on Formulating and Using QUBO Models](https://arxiv.org/abs/1811.11538?utm_source=chatgpt.com)

This is the paper I'd keep around for your **literature survey/project report**. It explains how constraints and optimization problems can be represented using QUBO. [arXiv](https://arxiv.org/abs/1811.11538?utm_source=chatgpt.com)

### 3. **D-Wave's QUBO explanation**

[D-Wave — QUBO / Binary Quadratic Model documentation](https://docs.dwavequantum.com/en/latest/shared/parameters.html?utm_source=chatgpt.com)

Useful when you want to understand what the actual `Q` matrix means and how the linear/quadratic coefficients represent the optimization problem. [D-Wave Documentation](https://docs.dwavequantum.com/en/latest/shared/parameters.html?utm_source=chatgpt.com)

### 4. **Quantum-inspired binary optimization**

[Quantum-Inspired Binary Optimization Algorithm for Representative Selection](https://arxiv.org/abs/2301.01836?utm_source=chatgpt.com)

This is closer to what **we're actually trying to do**: use quantum-inspired binary optimization for a practical selection problem. [arXiv](https://arxiv.org/abs/2301.01836?utm_source=chatgpt.com)

### 5. **Quantum-inspired evolutionary algorithms**

[Adaptive Quantum-inspired Differential Evolution Algorithm for 0-1 Knapsack](https://arxiv.org/abs/1101.0362?utm_source=chatgpt.com)

This is useful for understanding the idea of Q-bit representation and quantum-inspired search in a binary optimization problem. [arXiv](https://arxiv.org/abs/1101.0362?utm_source=chatgpt.com)

### 6. **If you want to understand the actual quantum side**

[IBM Quantum — QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm?utm_source=chatgpt.com)