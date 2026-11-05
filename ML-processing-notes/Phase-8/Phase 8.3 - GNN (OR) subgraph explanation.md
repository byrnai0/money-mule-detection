___

That's where we stop asking _"which features affected the score?"_ and start asking:

> **Which neighboring accounts/connections in the graph contributed to this account's prediction?**

That's the part that will make the explainability section substantially more graph-specific.

___

### Approach

Since the models are message-passing GNNs, we'll use a practical perturbation method:

1. Extract the target account's 2-hop subgraph.
2. Calculate its normal ensemble risk score.
3. Remove one neighboring node at a time.
4. Recalculate the target's risk score.
5. Measure:

```
neighbor impact =
baseline risk − risk after removing neighbor
```

6. Rank the neighbors by absolute impact.
7. Save the resulting subgraph explanation.

This is much easier to reproduce than jumping straight into `GNNExplainer`, and it works with the exact sampled models we've already trained.

Run:

`python -c "import torch; g=torch.load('data/graphs/training_graph.pt', map_location='cpu', weights_only=False)['data']; print(g); print('keys:', g.keys())"`

Expected Output:

```
Data(x=[515088, 26], edge_index=[2, 5078336], y=[515088], train_mask=[515088], val_mask=[515088], test_mask=[515088], num_nodes=515088, edge_label=[5078336], amount_received=[5078336], amount_paid=[5078336], timestamp=[5078336])
keys: ['test_mask', 'edge_label', 'timestamp', 'x', 'amount_received', 'y', 'edge_index', 'amount_paid', 'val_mask', 'train_mask', 'num_nodes']
```

___

We'll produce two levels of explanation:

**1. Neighbor influence**

```
Target account
├── Neighbor A   impact +0.21
├── Neighbor B   impact +0.08
├── Neighbor C   impact -0.03
└── Neighbor D   impact -0.01
```

**2. Transaction context**

For the highest-impact neighbors:

```
Target → Neighbor
Amount paid       ₹...
Amount received   ₹...
Timestamp         ...
Edge label        ...
```

One important detail: the graph has **5,078,336 edges**, so we should **not** repeatedly rebuild the entire graph for every neighbor. We'll work only with the target's local 2-hop subgraph, just like in phase 8.2.

___

Create `subgraph_explanation.py`

We're deliberately calling the result **neighbor influence**, not "causal contribution."

We're perturbing a neighbor's feature vector while keeping the graph connection itself. That tells us how much the model's output changes when that neighbor's learned information is removed, but it doesn't establish that the real-world account caused the target account to be risky.

### Findings:

For this target, the local graph contains **one direct non-self neighbor**.

Removing that neighbor's feature information caused the risk score to **increase**:

```
0.510683 → 0.564987
```

So this neighbor's information was **suppressing the model's risk prediction** by about `0.0543`.