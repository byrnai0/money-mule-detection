___

Now I'm going to visualize the **ranking behavior** of the final detector.

Will create:

```
Precision-Recall curve
ROC curve
```

The PR curve is the more important one for this project because our positive class is only around 1.23%.

Also marking the threshold value on the PR curve so we can see exactly where our current operating point sits.

Created `evaluation_curves.py`. Run it.

(matplotlib is needed)

2 files are created:

```
data/
└── graphs/
    └── evaluation/
        ├── precision_recall_curve.csv
        ├── precision_recall_curve.png
        ├── roc_curve.csv
        └── roc_curve.png
```

### Output:


```
Model ranking quality
→ PR-AUC = 0.308526
→ ROC-AUC = 0.915761

Operational threshold
→ threshold = 0.79
→ precision = 16.52%
→ recall = 46.77%
```

