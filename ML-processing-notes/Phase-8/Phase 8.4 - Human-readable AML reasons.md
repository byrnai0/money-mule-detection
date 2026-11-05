___

Here, We'll translate model outputs into cautious, evidence-based language such as _"contributed to the model's risk score"_ rather than saying _"this proves money laundering."_

For example:

```
Account: 10_800059840
Risk score: 0.5107
Risk level: MEDIUM

Model evidence:
• Receiving currency diversity contributed strongly to the
  predicted risk score.
• Incoming payment-format diversity reduced the predicted
  risk score.
• Incoming transaction amount behavior contributed to the
  predicted risk score.

Graph evidence:
• 1 direct non-self neighbor was identified in the 2-hop
  explanation subgraph.
• Neighbor 1467_800D10B40 had a risk-suppressing effect
  of 0.0543 on the prediction.

Transaction context:
• 1 connecting transaction was found.
• Amount paid: 34.98
• Amount received: 34.98
• Edge label: 0
```

Create `aml_explanation.py`

Because 8.4 consumes the outputs from 8.2/8.3, first make sure you've run both for the account:

```
python .\src\models\individual_explanation.py --account-id 10_800059840
```

```
python .\src\models\subgraph_explanation.py --account-id 10_800059840
```

Then:

```
python .\src\models\aml_explanation.py --account-id 10_800059840
```

It should create:

```
data/
└── graphs/
    └── evaluation/
        └── account_explanations/
            └── 10_800059840.md
```


___
### Expected output:

```
# Account Risk Explanation

  

**Account ID:** `10_800059840`

  

**Risk score:** `0.510683`

  

**Risk level:** `MEDIUM`

  

## Model-based evidence

  

- Receiving currency diversity was a strong positive contributor to the model's predicted risk score.

- Incoming payment-format diversity had a strong opposing effect on the model's predicted risk score.

- Average incoming transaction amount was a strong positive contributor to the model's predicted risk score.

- Outgoing transaction activity had a strong opposing effect on the model's predicted risk score.

- Diversity of receiver banks had a strong opposing effect on the model's predicted risk score.

  

## Graph-based evidence

  

- Neighbor 1467_800D10B40 reduced the model's predicted risk score when its feature information was removed from the local graph (score change -0.0543); 1 connecting transaction(s) were identified.

  

## Important interpretation

  

These statements describe how the trained model responded to feature and neighbor perturbations. They do not establish that the account engaged in illicit activity.

  

## Transaction context

  

### Neighbor `1467_800D10B40`

  

- Edge 1870666: paid=34.98, received=34.98, edge_label=0.0, timestamp=1662164
```

For this account, the final explanation says:

```
Risk score: 0.510683
Risk level: MEDIUM
```

The strongest model-level contributors were:

1. Receiving currency diversity — positive contribution
2. Incoming payment-format diversity — opposing contribution
3. Average incoming transaction amount — positive contribution
4. Outgoing transaction activity — opposing contribution
5. Receiver-bank diversity — opposing contribution

Graph-wise, the analyzed local subgraph found:

```
Neighbor: 1467_800D10B40
Transaction count: 1
Paid: 34.98
Received: 34.98
Edge label: 0
```

And the neighbor had a **−0.0543 model-score effect**, meaning its feature information was associated with a lower predicted risk score in this perturbation experiment.

