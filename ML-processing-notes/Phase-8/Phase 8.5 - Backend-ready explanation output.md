___

Formatting the md findings into JSON file for future Flask/FastAPI backend so that it can directly return the output in the frontend.

Create `backend_explanation.py`

Output will be saved at `\money_mule_detection\data\graphs\evaluation\backend_explanations.json`

___

this JSON is deliberately structured so my baby gurl KM can eventually get the FastAPI endpoint and can essentially do:

```
GET /account/{account_id}/explanation
        ↓
backend_explanations.json
        ↓
JSON response
        ↓
frontend
```

No Markdown parsing, no model loading in the frontend, and no need for the UI to understand PyTorch.

___

For this account specifically:

```
Account: 10_800059840
Risk:    0.510683
Level:   MEDIUM
```

The strongest feature perturbation was:

```
Receiving currency diversity
impact: +0.430371
```

and the strongest opposing feature was:

```
Incoming payment format diversity
impact: -0.259438
```

The graph explanation contains:

```
Neighbor: 1467_800D10B40
Impact:  -0.054305
Transaction: 34.98 paid / 34.98 received
```

___

Phase 8 artifacts are now essentially:

```
src/models/
├── feature_importance.py
├── individual_explanation.py
├── subgraph_explanation.py
├── aml_explanation.py
└── backend_explanation.py

data/graphs/evaluation/
├── global_feature_importance.csv
├── individual_explanations.csv
├── subgraph_explanations.csv
├── account_explanations/
└── backend_explanations.json
```

### One important project-level point

We now have a pretty solid explainability story:

**Global:** what features the overall model relies on.

**Local:** why a particular account received its score.

**Graph:** which neighboring accounts affected that prediction.

**Transaction:** what connections exist in the explanatory subgraph.

**API-ready:** all of it is available as structured JSON.
