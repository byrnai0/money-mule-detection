___

### Phase 3 — Graph Construction ✅

We converted:

```
Account → Node
Transaction → Directed Edge
```

Result:

```
515,088 nodes
5,078,336 edges
5,177 laundering edges
```

and serialized the graph into PyTorch Geometric format.

The account-level graph is particularly important because that is what makes this a **network/mule detection** project rather than just a generic transaction classifier. The roadmap explicitly makes accounts the nodes and transactions the directed edges.

___

Link to other MD files of the same phase:
[[Phase 3.1 - Data node and transaction creation]]
[[Phase 3.2 - Debugging and Installing PyTorch with CUDA support]]
