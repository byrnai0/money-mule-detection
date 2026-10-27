___

So, 
ACCOUNT = NODE
TRANSACTION = DIRECTED EDGE

We change the representation of the data from normal tabular to:

From
```
Account A
    |
    | ₹/€/$ amount
    ↓
Account B
```

To
```
A ───────────────→ B
        edge
```

with the transaction's information attached to the edge:
```
edge
├── amount_received
├── amount_paid
├── timestamp
├── currencies
├── payment_format
└── is_laundering
```

the complete graph should be represented using **PyTorch Geometric tensors**

The further flow is:
```
Full dataset
    ↓
PyTorch Geometric
    ↓
small suspicious subgraph
    ↓
NetworkX
    ↓
visualization / analysis
```

___

# 3.3 One improvement from our earlier builder

I'm going to make a slight change from the builder I gave you earlier.

We should **preserve categorical transaction information** rather than throw it away. But we don't need to turn it into ML tensors yet.

So Phase 3 should create:

```
 Structural graph data

edge_index
```

```
 Raw edge information


amount_received
amount_paid
timestamp
payment format
currencies
is_laundering
```


Then **Phase 4** converts these into proper numerical features.

___

After all these shits from [[Phase 3.2 - Debugging and Installing PyTorch with CUDA support]] are done and builder.py is ran, we shud get this output:

```
Graph nodes creation in process
===========================================================================
PHASE 3 — GRAPH CONSTRUCTION
===========================================================================

Input dataset:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\HI-Small_Trans_clean.csv

[1/7] Loading cleaned dataset...
Transactions loaded: 5,078,336

[2/7] Creating globally unique account IDs...

[3/7] Creating account-to-node mapping...
Unique accounts / nodes: 515,088

[4/7] Building edge_index...

[5/7] Preparing edge information...

[6/7] Creating PyTorch Geometric graph...

[7/7] Saving graph...

===========================================================================
PHASE 3 — GRAPH CONSTRUCTION COMPLETE
===========================================================================

Nodes:              515,088
Edges:              5,078,336
Laundering edges:   5,177
Edge index shape:   (2, 5078336)
Runtime:             11.92 seconds

Graph saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\graphs\ibm_aml_graph.pt
```

___

# Output:

## Phase 3 result

```
Nodes:             515,088
Edges:           5,078,336
Laundering edges:     5,177
Edge index:        (2, 5,078,336)
Runtime:              11.92 s
```

That means our fundamental graph invariant holds:

```
5,078,336 cleaned transactions
          ↓
5,078,336 graph edges
```

and:

```
5,177 laundering transactions
          ↓
5,177 laundering edges
```

___

# What I built

Conceptually, I have:

```
515,088 Accounts
      ↓
   GRAPH NODES
      ↑
      │
5,078,336 Transactions
      ↓
 DIRECTED EDGES
```

Each edge currently carries:

```
amount_received
amount_paid
timestamp
is_laundering
```

The categorical information such as:

```
payment_format
payment_currency
receiving_currency
```

___

So i created a file at root called `test_graph.py` to test the `.pt` that has been created. The output should look like this:

```
Graph loaded successfully
Nodes: 515088
Edges: 5078336
Laundering edges: 5177
Accounts: 515088
Edge index shape: (2, 5078336)
```

