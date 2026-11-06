___
# Phase 9 --- Experimental Methods, Results, and Findings

## Money Mule Network Detection in Financial Transactions

**Document purpose:** This document explains the Phase 9 experiments in
enough detail that another project teammate can understand what was
implemented, why it was tested, how to interpret the results, and what
conclusions are justified.

**Project context:** The project detects potential money-mule activity
using graph-based representations of financial accounts and transaction
behavior. Earlier phases built the data pipeline, graph, node features,
graph neural network (GNN) models, ensemble, and evaluation process.
Phase 9 investigated additional optimization and graph-learning methods
without assuming that a newer or more complex method would necessarily
outperform the existing baseline.

> **Important reporting rule:** Results below are separated by
> experiment and evaluation setup. A metric measured at a fixed
> threshold of 0.50 must not be treated as directly comparable to one
> measured at a validation-selected threshold. Validation results are
> used for model selection and threshold selection; the test set is
> reserved for final evaluation.

------------------------------------------------------------------------

## 1. Phase 9 at a glance

Phase 9 explored four experimental directions:

1.  **QUBO-based ensemble-weight optimization:** use a quadratic
    unconstrained binary optimization formulation, solved with classical
    simulated annealing, to select ensemble weights by minimizing
    validation Brier score.
2.  **Quantum-inspired evolutionary algorithm (QIEA) feature
    selection:** search for a smaller set of node features, then assess
    whether those features remain useful in the GNN/ensemble pipeline.
3.  **CP-decomposition-inspired GNN:** test a low-rank, multiplicative
    node-neighbor interaction architecture.
4.  **CP-style multiplicative graph pooling:** test a second
    architecture that aggregates projected neighborhood representations
    using an element-wise product.

The purpose was experimental comparison, not to replace the baseline at
any cost. The results show that the original ensemble remains a strong
reference model. The experiments provide useful findings about
calibration, feature selection, model behavior, and the limits of the
tested alternative architectures.

### Current overall conclusions

-   **QUBO:** produced feasible weights and improved test Brier score
    versus the saved comparison ensemble at threshold 0.50, but did not
    improve test PR-AUC or ROC-AUC. Its fixed-threshold F2 was slightly
    higher, while recall was slightly lower.
-   **QIEA:** selected 12 of 26 features in a screening pilot. The final
    saved hybrid ensemble achieved test PR-AUC 0.305162 and F2 0.342410
    at threshold 0.85, slightly below the baseline's reported
    threshold-selected PR-AUC 0.306027 and F2 0.349599. The QIEA
    operating point flagged fewer accounts and had higher precision, but
    lower recall.
-   **CP-inspired GNN:** below the baseline on test PR-AUC, ROC-AUC, and
    F2.
-   **CP-style pooling:** below the baseline on test PR-AUC, ROC-AUC,
    and F2.
-   **No quantum advantage was demonstrated.** The QUBO optimization was
    run using classical simulated annealing, and the QIEA was a
    quantum-inspired algorithm rather than evidence of execution on
    quantum hardware.

------------------------------------------------------------------------

## 2. Shared dataset and baseline context

The experiments use the existing graph artifact:

`data/graphs/training_graph.pt`

The graph payload is stored under `data["data"]` and contains the
PyTorch Geometric graph data.

### Graph and split information

  Item                                  Value
  ------------------------------- -----------
  Nodes                               515,088
  Edges                             5,078,336
  Node input features                      26
  Training nodes                      412,070
  Validation nodes                     51,509
  Test nodes                           51,509
  Positive examples in test set           635

The data is highly imbalanced: positive money-mule examples are rare
compared with negative examples. Consequently, accuracy alone would be
misleading. The experiments emphasize **PR-AUC**, **precision**,
**recall**, and **F2-score**, alongside ROC-AUC and, where relevant,
Brier score.

### What the metrics mean

-   **PR-AUC:** summarizes the precision-recall curve across thresholds.
    It is especially informative when the positive class is rare.
-   **ROC-AUC:** measures ranking quality across thresholds, considering
    true-positive and false-positive rates.
-   **Precision:** among accounts flagged as positive, the proportion
    that are actually positive.
-   **Recall:** among actual positive accounts, the proportion the model
    identifies.
-   **F1-score:** harmonic mean of precision and recall, weighting them
    equally.
-   **F2-score:** F-score that weights recall more heavily than
    precision. This is useful when missing suspicious accounts is
    considered particularly costly.
-   **Brier score:** mean squared error between predicted probabilities
    and binary labels. Lower values indicate better probabilistic
    accuracy/calibration overall.
-   **Flagged count:** number of accounts classified as positive at a
    particular threshold.

### Baseline ensemble

The existing ensemble combines predictions from GraphSAGE, GATv2, and
GIN. The saved PR-AUC-tuned weights are:

  Model         Baseline ensemble weight
  ----------- --------------------------
  GraphSAGE                         0.78
  GATv2                             0.19
  GIN                               0.03

The main baseline benchmark previously recorded at a validation-selected
threshold of 0.82 is:

  Metric               Baseline test result
  ------------------ ----------------------
  PR-AUC                           0.306027
  ROC-AUC                          0.914675
  Precision                        0.182394
  Recall                           0.453543
  F1-score                         0.260163
  F2-score                         0.349599
  Flagged accounts                    1,579

A separate saved QUBO report contains an existing-ensemble comparison at
threshold 0.50 with slightly different reported PR-AUC and ROC-AUC
values. This document preserves those values within the QUBO experiment
rather than silently mixing evaluation setups.

------------------------------------------------------------------------

# 3. Experiment A --- QUBO ensemble-weight optimization

## 3.1 Objective

The objective was to find ensemble weights that combine the three model
prediction streams while minimizing the validation Brier score.

The weights were constrained to be nonnegative and represented in 0.01
increments, with the weights summing to 1. This makes the ensemble a
convex weighted combination of the component models.

The optimization was formulated as a **quadratic unconstrained binary
optimization (QUBO)** problem. Binary variables encode discrete weight
units, and a penalty term discourages solutions that violate the
required total weight.

## 3.2 Solver and configuration

The saved report identifies the method as **QUBO with simulated
annealing**.

  Setting                                Value
  -------------------------------------- ----------------------------------
  Objective                              Validation Brier score
  Weight increment                       0.01
  Constraint                             Nonnegative weights summing to 1
  Component models                       GraphSAGE, GATv2, GIN
  Number of reads                        2,000
  Number of sweeps                       2,000
  Seed in primary report                 42
  Constraint penalty in primary report   0.01

The report explicitly warns that simulated annealing is heuristic and
does not guarantee a global optimum. Because this was classical
simulated annealing, it should not be described as execution on a
quantum computer.

## 3.3 Selected weights

  Model         Existing weight   QUBO weight
  ----------- ----------------- -------------
  GraphSAGE                0.78          0.68
  GATv2                    0.19          0.32
  GIN                      0.03          0.00

The optimizer shifted weight away from GraphSAGE toward GATv2 and
assigned zero weight to GIN. The result suggests that, under the chosen
objective and settings, GIN did not contribute enough to the optimized
probability combination to receive weight.

This does not prove that GIN is generally unhelpful; the conclusion
applies to this ensemble, dataset, objective, and experiment.

## 3.4 Results in the saved report

The report evaluates both ensembles at the fixed threshold of **0.50**.

### Validation results

  Metric               QUBO ensemble   Existing ensemble
  ------------------ --------------- -------------------
  Brier score ↓         **0.115021**            0.115961
  PR-AUC ↑                  0.354176        **0.358850**
  ROC-AUC ↑                 0.916093        **0.916273**
  Precision                 0.058690            0.057158
  Recall                    0.816038        **0.822327**
  F1-score                  0.109505            0.106887
  F2-score              **0.227891**            0.223619
  Flagged accounts             8,843               9,150

### Test results

  Metric               QUBO ensemble   Existing ensemble
  ------------------ --------------- -------------------
  Brier score ↓         **0.114019**            0.114922
  PR-AUC ↑                  0.307865        **0.308526**
  ROC-AUC ↑                 0.914909        **0.915761**
  Precision             **0.059634**            0.058385
  Recall                    0.831496        **0.839370**
  F1-score              **0.111287**            0.109177
  F2-score              **0.231701**            0.228383
  Flagged accounts             8,854               9,129

**Interpretation:** QUBO improved the Brier score and produced slightly
higher precision, F1, and F2 at threshold 0.50, while flagging fewer
accounts. However, PR-AUC, ROC-AUC, and recall were slightly lower. The
result is a trade-off, not an overall win across every metric.

The QUBO report's fixed-threshold metrics should not be compared as if
they were measured at the baseline's validation-selected threshold of
0.82. The large number of flagged accounts at threshold 0.50
demonstrates why threshold choice matters in this imbalanced dataset.

## 3.5 Robustness experiment

The file `data/graphs/evaluation/phase9_qubo/qubo_robustness.csv`
records runs with several penalty values and seeds.

Across the tested penalty settings of 0.01, 0.1, and 1.0, the solutions
generally assigned around 62--68% weight to GraphSAGE, 32--38% to GATv2,
and 0% to GIN. With penalty 0.001, the solutions varied much more,
including a run with 56% GATv2.

All recorded solutions were feasible and had weight units summing to
100.

This supports the limited conclusion that the two-model GraphSAGE/GATv2
mixture was repeatedly favored under many of the tested settings. It
does not establish global optimality or quantum advantage.

## 3.6 QUBO conclusion

QUBO was successfully formulated and solved using classical simulated
annealing. It produced feasible ensemble weights and improved Brier
score in the saved comparison. However, the test ranking metrics were
marginally lower than those of the existing ensemble, so the baseline
should remain the main reference model.

**Saved files:**

-   `data/graphs/evaluation/phase9_qubo/qubo_ensemble_report.json`
-   `data/graphs/evaluation/phase9_qubo/qubo_robustness.csv`
-   `data/graphs/evaluation/phase9_qubo/qubo_weights.csv`

------------------------------------------------------------------------

# 4. Experiment B --- QIEA feature selection

## 4.1 Objective

The goal was to investigate whether a quantum-inspired evolutionary
algorithm could select a smaller, useful subset of node features. A
smaller feature set could reduce input complexity and might improve
generalization or computational efficiency, but this must be tested
rather than assumed.

The search evaluated feature subsets using a class-weighted logistic
regression model as a **screening proxy**. The fitness objective was
validation PR-AUC.

The QIEA experiment is distinct from the QUBO experiment: QUBO optimized
ensemble weights, whereas QIEA searched over feature subsets.

## 4.2 Screening-pilot configuration

  Setting                    Value
  -------------------------- ------------------------------------
  Screening model            Class-weighted logistic regression
  Training sample            100,000
  Validation sample          30,000
  Population size            10
  Generations                8
  Minimum features           5
  Maximum features           22
  Random seed                42
  Unique subsets evaluated   73
  Selected features          12

The pilot report warns that these are screening-proxy results and should
not be interpreted as final GNN performance.

## 4.3 Selected features

The pilot selected these 12 features:

1.  `log1p_in_transaction_count`
2.  `log1p_unique_senders`
3.  `log1p_incoming_active_days`
4.  `log1p_active_days`
5.  `log1p_incoming_payment_format_diversity`
6.  `log1p_receiving_currency_diversity`
7.  `log1p_payment_currency_diversity`
8.  `incoming_currency_abs_z_mean`
9.  `outgoing_log_amount_mean`
10. `outgoing_log_amount_std`
11. `outgoing_log_amount_max`
12. `outgoing_currency_abs_z_max`

Broadly, these features represent account activity, sender diversity,
transaction-format and currency diversity, and outgoing transaction
amount patterns. They are plausible signals for unusual financial
activity, but feature selection alone does not prove that each feature
is causally or independently informative.

## 4.4 Screening-pilot results

  Metric                All 26 features   QIEA-selected 12 features
  ------------------- ----------------- ---------------------------
  Validation PR-AUC            0.074061                **0.080913**
  Validation F2            **0.141743**                    0.133030

The selected subset increased pilot PR-AUC from 0.074061 to 0.080913,
approximately a 9.25% relative increase. However, pilot F2 decreased
from 0.141743 to 0.133030.

This indicates that the subset improved the pilot's ranking metric but
did not improve F2 at the pilot's operating point. The pilot's metrics
are not directly comparable to the final GNN test metrics because they
came from a different model and sampled data.

## 4.5 Final hybrid ensemble result

The saved archive is:

`data/graphs/evaluation/phase9_qiea/hybrid_ensemble_result.npz`

It contains six ensemble weights, a selected threshold, validation
metrics, test labels, and test probabilities.

Saved ensemble weights:

`[0.76, 0.19, 0.00, 0.03, 0.00, 0.02]`

The archive's available output does not itself provide a named mapping
for all six positions. Therefore, the six weights should not be assigned
model names in the report until the code that constructs the hybrid
ensemble confirms their order.

### Saved validation values

  Metric                    Value
  -------------------- ----------
  Validation PR-AUC      0.356874
  Validation F2          0.370176
  Selected threshold         0.85

### Test results at threshold 0.85

The test metrics were calculated from the saved `test_labels` and
`test_probabilities`.

  Metric                         Result
  -------------------------- ----------
  PR-AUC                       0.305162
  ROC-AUC                      0.914812
  Precision                    0.217280
  Recall                       0.400000
  F1-score                     0.281596
  F2-score                     0.342410
  Flagged accounts                1,169
  Actual positive examples          635
  True positives                    254
  False positives                   915
  True negatives                 49,959
  False negatives                   381

At this operating point, precision is higher and fewer accounts are
flagged than in the baseline evaluation. Recall is lower: the model
identified 254 of the 635 positive test examples and missed 381.

Compared with the previously recorded baseline at threshold 0.82, QIEA's
test PR-AUC (0.305162) and F2 (0.342410) are slightly lower than the
baseline's 0.306027 and 0.349599. Because the thresholds differ and were
selected independently on validation data, this is a comparison of the
reported operating points rather than a controlled, threshold-matched
experiment.

## 4.6 QIEA conclusion

QIEA demonstrated a practical way to screen feature subsets and produced
a 12-feature candidate set. The final saved hybrid ensemble was
competitive with the baseline but did not surpass it on test PR-AUC or
F2. Its higher precision and lower alert count may be useful if reducing
investigation workload is a priority, but that trade-off comes with
lower recall.

Do not claim that QIEA definitively improves the final GNN. A stronger
conclusion would require a controlled comparison, multiple seeds, and a
clear mapping of the six ensemble weights to their component predictors.

**Saved files include:**

-   `data/graphs/evaluation/phase9_qiea/qiea_pilot_report.json`
-   `data/graphs/evaluation/phase9_qiea/qiea_selected_features.csv`
-   `data/graphs/evaluation/phase9_qiea/feature_mapping.csv`
-   `data/graphs/evaluation/phase9_qiea/qiea_history.csv`
-   `data/graphs/evaluation/phase9_qiea/stability/feature_selection_frequency.csv`
-   `data/graphs/evaluation/phase9_qiea/hybrid_ensemble_result.npz`

Additional stability outputs and per-model validation prediction
archives are present in the QIEA evaluation directory.

------------------------------------------------------------------------

# 5. Experiment C --- CP-decomposition-inspired GNN

## 5.1 Objective

This experiment tested a low-rank node-neighbor interaction architecture
inspired by CP-style factorization. The intention was to investigate
whether multiplicative interactions between projected node and
neighborhood representations could improve the graph model.

This architecture is separate from the CP-style pooling model described
in the next section. It is also an adaptation, not a full reproduction
of the referenced high-order pooling paper.

## 5.2 Implementation

Files:

-   `src/models/cp_gnn.py`
-   `src/models/train_cp_gnn_sampled.py`
-   `src/models/evaluate_cp_gnn_threshold.py`

Checkpoint:

`data/graphs/models/sampled/cp_gnn_sampled_model.pt`

The model was trained with sampled graph neighborhoods and evaluated on
the existing train/validation/test split. A threshold was selected on
validation data by maximizing F2 and then applied to the test set.

## 5.3 Results

### Test results at threshold 0.50

  Metric                 Result
  ------------------ ----------
  PR-AUC               0.280574
  ROC-AUC              0.895055
  Precision            0.057945
  Recall               0.763780
  F1-score             0.107718
  F2-score             0.222273
  Flagged accounts       10,395

### Test results at validation-selected threshold 0.807859

  Metric                 Result
  ------------------ ----------
  PR-AUC               0.281408
  ROC-AUC              0.896621
  Precision            0.160724
  Recall               0.447244
  F1-score             0.236470
  F2-score             0.329696
  Flagged accounts        1,767

The selected threshold substantially reduced the number of alerts.
Nevertheless, the model's test PR-AUC, ROC-AUC, and F2 remained below
the original ensemble's reported results.

## 5.4 Conclusion

The CP-inspired GNN executed and produced usable predictions, but it did
not outperform the existing ensemble. It is retained as an exploratory
model and not selected as the final detection model.

**Saved evaluation file:**
`data/graphs/evaluation/phase9_cp_gnn_threshold.json`

------------------------------------------------------------------------

# 6. Experiment D --- CP-style multiplicative graph pooling

## 6.1 Objective

The purpose was to explore an alternative graph aggregation operation
based on element-wise multiplication of projected neighborhood
representations.

The motivating reference was:

**High-Order Pooling for Graph Neural Networks with Tensor
Decomposition**\
Chenqing Hua, Guillaume Rabusseau, and Jian Tang, NeurIPS 2022.

-   Official NeurIPS page:
    https://papers.neurips.cc/paper_files/paper/2022/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html
-   arXiv: https://arxiv.org/abs/2205.11691

The implemented architecture is CP-decomposition-inspired; it is not a
full reproduction of the paper's tGNN architecture.

## 6.2 Architecture

Implementation:

`src/models/cp_pooling_gnn.py`

The model defines `CPLayer` and `CPPoolingGNN`.

At a high level, the layer:

1.  Appends a constant feature to each node representation.
2.  Projects the augmented representation into a lower-dimensional
    space.
3.  Applies a `tanh` activation.
4.  Aggregates projected neighborhood representations using element-wise
    product through a product reduction.
5.  Applies a learned output projection.

The network contains two such layers. It applies ReLU to the hidden
representation, uses dropout, and produces two final class logits.
Class-weighted cross-entropy is used for training.

The central difference from mean aggregation is that neighborhood
messages are combined by multiplication rather than averaging. This can
model interactions between projected components, but it can also be
sensitive to small values, signs, and neighborhood size.

## 6.3 Implementation tests

Files:

-   `src/models/test_cp_pooling_gnn.py`
-   `src/models/test_cp_pooling_sampled.py`

The model-level test passed on CUDA with:

-   Output shape `(20, 2)`
-   Finite logits
-   Working gradient computation

The sampled-batch test passed using a real sampled graph batch with
approximately 639 sampled nodes, 1,023 edges, and 64 seed nodes. It
produced output shape `(64, 2)` and completed the backward pass.

These tests established basic execution and training compatibility, not
performance superiority.

## 6.4 Training setup

Trainer:

`src/models/train_cp_pooling_sampled.py`

The model used the existing sampled training setup:

-   Neighbor fanouts: `[15, 10]`
-   Batch size: 1,024
-   Maximum epochs: 20
-   Learning rate: 0.001
-   Weight decay: 0.0001
-   Class-weighted cross-entropy
-   Best checkpoint selected by validation PR-AUC

Best epoch: 19\
Best validation PR-AUC: 0.221457

Checkpoint:

`data/graphs/models/sampled/cp_pooling_sampled_model.pt`

## 6.5 Results at threshold 0.50

  Metric                 Result
  ------------------ ----------
  PR-AUC               0.177991
  ROC-AUC              0.875354
  Precision            0.046657
  Recall               0.763780
  F1-score             0.087942
  F2-score             0.187476
  Flagged accounts       10,395

The model achieved high recall at the default threshold but had very low
precision and generated many alerts.

## 6.6 Validation-based threshold selection

The threshold evaluation script searched thresholds from 0.01 to 0.99
and selected the threshold maximizing validation F2.

Selected threshold: `0.830626`

Validation metrics at that threshold:

  Metric                 Result
  ------------------ ----------
  PR-AUC               0.218241
  ROC-AUC              0.890034
  Precision            0.159873
  Recall               0.394654
  F1-score             0.227561
  F2-score             0.305056
  Flagged accounts        1,570

## 6.7 Final test results

The selected threshold was applied unchanged to the test set.

  Metric                 Result
  ------------------ ----------
  PR-AUC               0.172370
  ROC-AUC              0.875490
  Precision            0.138854
  Recall               0.343307
  F1-score             0.197732
  F2-score             0.265207
  Flagged accounts        1,570

## 6.8 Comparison with baseline

  Metric            Original ensemble  CP-style pooling GNN
  ----------------  -----------------  --------------------
  Test PR-AUC                0.306027              0.172370
  Test ROC-AUC               0.914675              0.875490
  Test precision             0.182394              0.138854
  Test recall                0.453543              0.343307
  Test F1-score              0.260163              0.197732
  Test F2-score              0.349599              0.265207
  Flagged accounts              1,579                 1,570

The models were evaluated at similar alert volumes, with thresholds
selected on validation data. The CP-style pooling model performed worse
across all listed metrics.

## 6.9 Interpretation and limitations

Potential reasons for the weaker result include the sensitivity of
multiplicative aggregation to small values and signs, variation in
neighborhood size, optimization difficulty, and architecture-specific
hyperparameter choices. These are hypotheses, not proven causes.

The experiment used one reported run and did not establish multi-seed
performance. It should not be used to claim that all CP-based graph
pooling methods are ineffective. The defensible conclusion is narrower:
this particular CP-inspired implementation did not outperform the
project's existing ensemble under the tested configuration.

**Saved files:**

-   `src/models/cp_pooling_gnn.py`
-   `src/models/test_cp_pooling_gnn.py`
-   `src/models/test_cp_pooling_sampled.py`
-   `src/models/train_cp_pooling_sampled.py`
-   `src/models/evaluate_cp_pooling_threshold.py`
-   `data/graphs/models/sampled/cp_pooling_sampled_model.pt`
-   `data/graphs/evaluation/phase9_cp_pooling_results.json`
-   `data/graphs/evaluation/phase9_cp_pooling_threshold.json`

------------------------------------------------------------------------

# 7. Consolidated comparison

The following table collects the principal reported test results.
Thresholds and evaluation setups are shown because they affect
precision, recall, F-scores, and alert counts.

  -----------------------------------------------------------------------------------------
  Experiment       Threshold used    Test PR-AUC   Test ROC-AUC        Test F2      Flagged
                    for displayed                                                  accounts
                  operating-point                                              
                          metrics                                              
  ------------- ----------------- -------------- -------------- -------------- ------------
  Original                   0.82   **0.306027**   **0.914675**   **0.349599**        1,579
  ensemble                                                                     

  QUBO ensemble              0.50     0.307865\*     0.914909\*     0.231701\*        8,854

  QIEA hybrid                0.85       0.305162       0.914812       0.342410        1,169
  ensemble                                                                     

  CP-inspired            0.807859       0.281408       0.896621       0.329696        1,767
  GNN                                                                          

  CP-style               0.830626       0.172370       0.875490       0.265207        1,570
  pooling GNN                                                                  
  -----------------------------------------------------------------------------------------

`*` QUBO PR-AUC and ROC-AUC are threshold-independent ranking metrics in
principle, but the values here come from the saved QUBO report's own
evaluation run and should be compared cautiously with results from other
evaluation scripts. Its precision, recall, F2, and alert count
definitely reflect the fixed 0.50 threshold.

### How to read this table

-   The original ensemble remains the reference because it has the
    strongest threshold-selected F2 among the main baseline and
    alternative GNN results reported here.
-   QUBO's fixed-threshold operating point has lower F2 than the
    threshold-selected baseline, but its Brier score is better in the
    QUBO report.
-   QIEA has a similar test PR-AUC to the baseline and a lower F2 at its
    saved threshold, while flagging fewer accounts.
-   Both CP-inspired models fall below the baseline on the principal
    reported test metrics.
-   These findings do not prove that the methods cannot work under any
    configuration; they describe the outcomes of the experiments
    actually run.

For a strict apples-to-apples comparison, all models should be evaluated
using the same data splits and metric implementation, with thresholds
selected exclusively on validation data according to the same rule. The
current report preserves the saved experiment results and flags
differences rather than pretending all rows came from an identical
protocol.

------------------------------------------------------------------------

# 8. Overall Phase 9 findings

Phase 9 investigated optimization and alternative graph-learning methods
around the existing money-mule detection pipeline.

## What was achieved

1.  A QUBO formulation was used to optimize ensemble weights under a
    Brier-score objective.
2.  Multiple QUBO seeds and penalty values were tested for robustness.
3.  A QIEA screening pilot reduced the candidate feature set from 26 to
    12.
4.  A final hybrid ensemble prediction archive was evaluated using its
    saved threshold.
5.  A CP-inspired GNN was implemented, trained, and evaluated.
6.  A CP-style multiplicative pooling GNN was implemented, tested with
    sampled graph batches, trained, and evaluated.
7.  Results were compared against the existing ensemble, including
    negative experimental findings.

## What was not established

-   No quantum hardware advantage was demonstrated.
-   QUBO did not improve all test metrics over the existing ensemble.
-   The QIEA pilot's PR-AUC improvement did not translate into a
    demonstrated improvement in final test PR-AUC or F2.
-   Neither CP-inspired architecture outperformed the original ensemble
    in the reported tests.
-   The experiments do not establish that these methods are universally
    inferior; the conclusions apply to the implementations and
    configurations evaluated.

## Final Phase 9 decision

Keep the original ensemble as the primary reference model. Retain QUBO,
QIEA, and CP experiment results as documented research investigations.
Do not replace the baseline based solely on novelty or on a single
favorable metric.

The practical value of Phase 9 is that the project tested several
plausible alternatives, measured their behavior, identified trade-offs,
and avoided claiming improvements unsupported by test results.

------------------------------------------------------------------------

# 9. Reproducibility checklist

Before using this report as the final submission record, verify the
following in the project repository:

-   [ ] Confirm all referenced source files and checkpoints exist.
-   [ ] Confirm the six positions in the QIEA hybrid ensemble weight
    vector and document their model mapping.
-   [ ] Preserve the QUBO fixed-threshold results separately from the
    baseline's validation-threshold results.
-   [ ] Confirm the validation threshold-selection method for each model
    from its evaluation script.
-   [ ] Keep test data out of model and threshold selection.
-   [ ] If making stronger performance claims, rerun comparisons with a
    consistent evaluation script and multiple random seeds.
-   [ ] Record the versions of Python, PyTorch, and PyTorch Geometric
    used for the final reproducibility package.
-   [ ] Ensure the final project report distinguishes classical
    simulated annealing from quantum computation and quantum-inspired
    methods from execution on quantum hardware.

------------------------------------------------------------------------

# 10. Artifact index

## QUBO

-   `data/graphs/evaluation/phase9_qubo/qubo_ensemble_report.json`
-   `data/graphs/evaluation/phase9_qubo/qubo_robustness.csv`
-   `data/graphs/evaluation/phase9_qubo/qubo_weights.csv`

## QIEA

-   `data/graphs/evaluation/phase9_qiea/qiea_pilot_report.json`
-   `data/graphs/evaluation/phase9_qiea/qiea_selected_features.csv`
-   `data/graphs/evaluation/phase9_qiea/feature_mapping.csv`
-   `data/graphs/evaluation/phase9_qiea/qiea_history.csv`
-   `data/graphs/evaluation/phase9_qiea/stability/feature_selection_frequency.csv`
-   `data/graphs/evaluation/phase9_qiea/hybrid_ensemble_result.npz`
-   Additional per-model validation prediction archives and stability
    outputs are stored under `data/graphs/evaluation/phase9_qiea/`.

## CP-inspired GNN

-   `src/models/cp_gnn.py`
-   `src/models/train_cp_gnn_sampled.py`
-   `src/models/evaluate_cp_gnn_threshold.py`
-   `data/graphs/evaluation/phase9_cp_gnn_threshold.json`
-   `data/graphs/models/sampled/cp_gnn_sampled_model.pt`

## CP-style multiplicative pooling

-   `src/models/cp_pooling_gnn.py`
-   `src/models/test_cp_pooling_gnn.py`
-   `src/models/test_cp_pooling_sampled.py`
-   `src/models/train_cp_pooling_sampled.py`
-   `src/models/evaluate_cp_pooling_threshold.py`
-   `data/graphs/evaluation/phase9_cp_pooling_results.json`
-   `data/graphs/evaluation/phase9_cp_pooling_threshold.json`
-   `data/graphs/models/sampled/cp_pooling_sampled_model.pt`

------------------------------------------------------------------------

**End of Phase 9 experimental report.**
