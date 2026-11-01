___

Here:

> **How are the model's risk scores distributed for legitimate versus illicit accounts?**
> **and** 
> **Does the tuned ensemble give genuinely illicit accounts higher risk scores, and where do the two populations overlap?**

and creating distributions for:

```
legitimate accounts
vs
illicit accounts
```

and mark:
threshold = 0.79

This is useful because we can see whether the model naturally separates the two groups or whether they heavily overlap.

We'll also calculate simple score statistics such as:
min
median
mean
max
for both groups.

We'll produce:

```
data/graphs/evaluation/
├── risk_score_distribution.png
└── risk_score_statistics.csv
```

Created `risk_score_analysis.py`. Run it.

### Output:

```
===========================================================================
PHASE 7.4 — RISK SCORE ANALYSIS
===========================================================================

[1/4] Loading test risk scores...
Test accounts: 51,509

[2/4] Separating legitimate and illicit accounts...

LEGITIMATE ACCOUNTS
count     : 50,874
mean      : 0.237377
median    : 0.158196
min       : 0.002837
max       : 0.999886
std       : 0.241821
p90       : 0.617345
p95       : 0.721763
p99       : 0.886769

ILLICIT ACCOUNTS
count     : 635
mean      : 0.731986
median    : 0.764811
min       : 0.056275
max       : 0.999973
std       : 0.225792
p90       : 0.986549
p95       : 0.996824
p99       : 0.999950

[3/4] Checking score separation at threshold...

Threshold: 0.79
Legitimate >= threshold: 1,501
Illicit >= threshold:     297
Legitimate < threshold:  49,373
Illicit < threshold:      338
```

___

The separation is strong:

```
                 Legitimate    Illicit
Mean score          0.237       0.732
Median              0.158       0.765
P90                 0.617       0.987
P95                 0.722       0.997
P99                 0.887       1.000
```

So illicit accounts are very clearly shifted toward higher risk scores.

**legitimate accounts are heavily concentrated near the low-score region, while illicit accounts are concentrated toward the high-score region.**

Some legitimate accounts get extremely high scores:

```
legitimate max = 0.999886
```

and some illicit accounts get relatively low scores:

```
illicit min = 0.056275
```

So the model **does not have perfect class separation**.

That is exactly what produces:

```
FP = 1,501
FN = 338
```

at the `0.79` threshold.

