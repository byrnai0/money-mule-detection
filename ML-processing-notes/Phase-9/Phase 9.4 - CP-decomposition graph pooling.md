___
## 1. Objective

The objective of this experiment was to investigate whether CP-decomposition-inspired multiplicative pooling could improve money mule detection in our graph neural network.

Our existing system uses graph neural networks to learn account representations from the transaction graph. The CP-style pooling experiment explored a different way of aggregating information from neighboring nodes: instead of relying only on conventional aggregation, it combines projected node and neighborhood representations through element-wise multiplication.

The experiment was part of Phase 9, which investigates quantum-inspired and alternative graph-learning techniques for the Money Mule Network Detection project.

## 2. Research Reference

The experiment was motivated by the paper:

**High-Order Pooling for Graph Neural Networks with Tensor Decomposition**

Authors: Chenqing Hua, Guillaume Rabusseau, and Jian Tang  
Conference: NeurIPS 2022

- Official NeurIPS paper page: [https://papers.neurips.cc/paper_files/paper/2022/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html](https://papers.neurips.cc/paper_files/paper/2022/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html)
    
- arXiv: [https://arxiv.org/abs/2205.11691](https://arxiv.org/abs/2205.11691)
    

The paper investigates high-order pooling methods for graph neural networks using tensor decomposition.

**Implementation note:** Our implementation was a CP-decomposition-inspired adaptation, not a complete reproduction of the paper's architecture. The results should therefore be described as results from our own CP-style pooling experiment, rather than results from the original paper's model.

## 3. Dataset and Experimental Setup

The experiment used the existing training graph:

`data/graphs/training_graph.pt`

The graph contained:

- Number of nodes: 515,088
    
- Number of edges: 5,078,336
    
- Number of input features per node: 26
    
- Training nodes: 412,070
    
- Validation nodes: 51,509
    
- Test nodes: 51,509
    
- Positive examples in the test set: 635
    

The dataset is highly imbalanced, with substantially more negative examples than positive money mule examples.

The model was trained using sampled neighborhood batches rather than loading the entire graph neighborhood into each training step.

### Main training configuration

- Framework: PyTorch and PyTorch Geometric
    
- Training approach: Neighbor sampling
    
- Neighbor fanouts: [15, 10]
    
- Batch size: 1,024
    
- Maximum epochs: 20
    
- Learning rate: 0.001
    
- Weight decay: 0.0001
    
- Loss: Class-weighted cross-entropy
    
- Model selection: Best validation PR-AUC
    
- Evaluation: Validation and test metrics
    

The implementation used the existing graph, train/validation/test splits, and sampled training workflow to keep the experiment consistent with the project's graph-learning pipeline.

## 4. Model Architecture

The implementation was created in:

`src/models/cp_pooling_gnn.py`

The model contains two CP-style pooling layers, implemented using the `CPLayer` and `CPPoolingGNN` classes.

### 4.1 Projecting node features

Each node's feature representation is augmented with a constant feature. A learned projection matrix maps the augmented features into a lower-dimensional representation.

A hyperbolic tangent activation is applied to the projected representation.

The purpose of this projection is to produce a compact representation that can participate in multiplicative interactions.

### 4.2 Neighborhood aggregation through multiplication

The implementation aggregates projected neighbor representations using an element-wise product.

Unlike conventional mean aggregation, which averages neighbor representations, multiplicative aggregation combines their projected components through a product operation.

This can represent interactions in which the combined neighborhood signal differs from a simple average. However, multiplication can also be sensitive to small values, signs, and the number of contributing neighbors.

The implementation uses a product reduction over neighborhood messages.

### 4.3 Output projection

The aggregated representation is passed through another learned linear projection.

The two-layer network applies ReLU to the hidden representation and uses dropout for regularization. The final layer produces two class logits for classification.

Cross-entropy loss is used during training, with class weights to account for the class imbalance.

### Simplified processing flow

Input node features  
→ Augmented feature representation  
→ Learned low-rank projection  
→ Multiplicative neighborhood aggregation  
→ Output projection  
→ Second CP-style pooling layer  
→ Classification logits  
→ Class-weighted cross-entropy loss

## 5. Implementation and Validation

The experiment used these files:

- `src/models/cp_pooling_gnn.py` — model architecture and CP-style pooling layer.
    
- `src/models/test_cp_pooling_gnn.py` — basic model test.
    
- `src/models/test_cp_pooling_sampled.py` — sampled-batch integration test.
    
- `src/models/train_cp_pooling_sampled.py` — model training.
    
- `src/models/evaluate_cp_pooling_threshold.py` — validation-based threshold selection and test evaluation.
    

### Tests performed

**Model-level test**

The model was tested using synthetic input data. It produced an output of shape `(20, 2)`, with finite logits and working gradient computation.

**Sampled-batch test**

The architecture was tested with an actual sampled batch from the graph. One successful test used approximately:

- 639 sampled nodes
    
- 1,023 sampled edges
    
- 64 seed nodes
    

The model produced an output of shape `(64, 2)`, and the backward pass completed successfully.

These tests established that the architecture could execute, process sampled graph neighborhoods, produce classification logits, and participate in gradient-based training.

They did not establish that the architecture was more accurate than the baseline.

## 6. Training Results

The training script selected the checkpoint with the best validation PR-AUC.

- Best epoch: 19
    
- Best validation PR-AUC: 0.221457
    

The first evaluation used a classification threshold of 0.50.

### Test results at threshold 0.50

|Metric|Result|
|---|---|
|PR-AUC|0.177991|
|ROC-AUC|0.875354|
|Precision|0.046657|
|Recall|0.763780|
|F1-score|0.087942|
|F2-score|0.187476|
|Flagged accounts|10,395|

The recall was high, but precision was very low. The model flagged a large number of accounts, most of which were negative examples.

This showed that the default threshold was not a useful operating point for controlling false-positive alerts.

## 7. Validation-Based Threshold Selection

To evaluate the model at a more practical operating point, the threshold evaluation script searched thresholds from 0.01 to 0.99 and selected the threshold that maximized validation F2-score.

The selected threshold was:

`0.830626`

### Validation metrics at the selected threshold

|Metric|Result|
|---|---|
|PR-AUC|0.218241|
|ROC-AUC|0.890034|
|Precision|0.159873|
|Recall|0.394654|
|F1-score|0.227561|
|F2-score|0.305056|
|Flagged accounts|1,570|

The threshold substantially reduced the number of flagged accounts and improved precision. Recall decreased because the model became more selective.

The threshold was selected using validation data and then applied unchanged to the test set.

## 8. Final Test Results

### Test metrics at threshold 0.830626

|Metric|Result|
|---|---|
|PR-AUC|0.172370|
|ROC-AUC|0.875490|
|Precision|0.138854|
|Recall|0.343307|
|F1-score|0.197732|
|F2-score|0.265207|
|Flagged accounts|1,570|

The test results were weaker than the validation results. This indicates that the validation-selected operating point did not transfer perfectly to unseen test data.

The experiment did not demonstrate a performance improvement over the original ensemble.

## 9. Comparison with the Original Ensemble

The original ensemble was evaluated using a validation-selected threshold of 0.82.

|Metric|Original ensemble|CP-style pooling GNN|
|---|---|---|
|Test PR-AUC|0.306027|0.172370|
|Test ROC-AUC|0.914675|0.875490|
|Test precision|0.182394|0.138854|
|Test recall|0.453543|0.343307|
|Test F1-score|0.260163|0.197732|
|Test F2-score|0.349599|0.265207|
|Flagged accounts|1,579|1,570|

The models were evaluated at similar alert volumes after their thresholds were selected on validation data. The CP-style pooling model performed worse across all listed classification and ranking metrics.

The difference in PR-AUC is particularly important because PR-AUC measures the precision-recall trade-off across classification thresholds. It is useful for evaluating a model on a highly imbalanced dataset.

## 10. Interpretation of the Results

The experiment successfully implemented and tested a CP-decomposition-inspired multiplicative pooling architecture within the project's sampled graph-learning workflow.

However, the experiment did not establish that multiplicative pooling improves money mule detection in this dataset.

Possible explanations for the weaker performance include:

1. **Sensitivity of multiplicative aggregation:** Element-wise products can shrink representations when projected values have small magnitudes or amplify differences between neighbor representations.
    
2. **Neighborhood-size effects:** The product aggregation can behave differently as the number of contributing neighbors changes.
    
3. **Optimization difficulty:** The multiplicative interaction may be harder to optimize than the aggregation mechanisms used by the baseline models.
    
4. **Architecture and hyperparameter choices:** The implementation was an experimental adaptation, and its rank, projections, activation functions, and training configuration may affect its performance.
    
5. **Dataset characteristics:** The graph's extreme class imbalance and the specific patterns in the money mule dataset may favor other architectures.
    

These are possible explanations, not experimentally proven causes. Further controlled experiments would be required to determine which factors contributed to the performance difference.

## 11. Limitations

- The implementation was CP-decomposition-inspired, not a full reproduction of the referenced paper.
    
- The experiment used one reported training run; performance across multiple random seeds was not established.
    
- The experiment did not demonstrate an improvement over the original ensemble.
    
- The threshold was selected using validation F2, so the resulting operating point prioritizes recall more than precision.
    
- The test results were weaker than the validation results.
    
- The experiment does not establish a quantum advantage or prove that CP-style pooling is generally ineffective.
    

## 12. Final Conclusion

The CP-decomposition-inspired graph pooling experiment explored multiplicative neighborhood interactions as an alternative to conventional graph aggregation.

The model was implemented, tested on synthetic data, integrated with sampled graph batches, trained, and evaluated using validation-based threshold selection. Its final test PR-AUC was 0.172370 and its F2-score was 0.265207, both below the original ensemble's corresponding results of 0.306027 and 0.349599.

Therefore, the CP-style pooling architecture was retained as an exploratory experiment rather than adopted as the final detection model.

The main outcome was methodological: the project tested an alternative graph-learning mechanism and established that, under the current implementation and training configuration, it did not outperform the existing ensemble. Reporting this result transparently helps justify retaining the baseline while documenting the alternatives investigated in Phase 9.

## 13. Output Artifacts

The following implementation and result files were produced:

- `src/models/cp_pooling_gnn.py`
    
- `src/models/test_cp_pooling_gnn.py`
    
- `src/models/test_cp_pooling_sampled.py`
    
- `src/models/train_cp_pooling_sampled.py`
    
- `src/models/evaluate_cp_pooling_threshold.py`
    
- `data/graphs/models/sampled/cp_pooling_sampled_model.pt`
    
- `data/graphs/evaluation/phase9_cp_pooling_results.json`
    
- `data/graphs/evaluation/phase9_cp_pooling_threshold.json`
    

These files provide the implementation, trained checkpoint, and recorded evaluation results for the experiment.