___

## Goal

Before building the graph, verify that the raw dataset is structurally valid.

### Validation performed

- Schema validation
- Missing-value check
- Duplicate check
- Label validation
- Amount validation
- Account-ID validation
- Self-transaction check
- Timestamp validation
- Class-distribution check

## Results

### Schema

```
Missing columns: None
Missing values:  None
```

### Labels

```
Unique labels: [0, 1]
Invalid labels: []
```

### Amounts

```
Non-numeric received: 0
Negative received:    0
Non-numeric paid:     0
Negative paid:        0
```

### Account IDs

```
Empty source IDs:      0
Empty destination IDs: 0
```

### Timestamps

```
Null timestamps: 0
```

The dataset contains many rows that are not chronologically ordered in the CSV. This is not automatically an error; chronological ordering will be handled when we build the temporal graph.

### Duplicates

Initial validation showed:

```
Exact duplicate records: 9
Physical duplicate rows: 18
```

These are **9 identical transaction records appearing twice**.

All 9 duplicated transaction records have:

```
Is Laundering = 0
```

Decision:

```
Remove one copy of each exact duplicate
```

Expected processed dataset:

```
5,078,345
-
9
=
5,078,336 transactions
```

The raw CSV should remain untouched. Cleaning should create a separate processed dataset.

### Self-transactions

```
591,212
```

Decision:

**KEEP for now.**

A self-transaction has:

```
from_account_id == to_account_id
```

We have not established that these are invalid records, so they should not be blindly deleted. Their treatment will be decided after understanding the graph structure.

### Class imbalance

```
Normal:      5,073,168
Laundering:      5,177

Laundering ratio:
0.101943%
```

This is a major project consideration.

Because laundering transactions are extremely rare:

- accuracy cannot be our main metric
- training must account for class imbalance
- precision / recall / F1 become important
- PR-AUC will be useful
- train/test splitting must avoid leakage

___

When cleaning the data, the output is this:


```
===========================================================================
IBM AML DATA VALIDATION
===========================================================================

Dataset:
data/raw/HI-Small_Trans.csv

Loading dataset...
This may take a while because the CSV contains ~5 million rows.


Dataset loaded successfully in 7.86 seconds.
Memory usage: 2.18 GB

[1/12] Checking schema...
[2/12] Checking dataframe shape...
[3/12] Checking missing values...
[4/12] Checking exact duplicate rows...
[5/12] Checking duplicate transaction records...
[6/12] Checking labels...
[7/12] Checking data types...
[8/12] Checking transaction amounts...
[9/12] Checking self-transactions...
[10/12] Checking account IDs...
[11/12] Checking timestamps...
[12/12] Checking class distribution...

Validation finished in 12.57 seconds.


===========================================================================
IBM AML DATA VALIDATION REPORT
===========================================================================

DATASET
Rows:                5,078,345
Columns:             13

SCHEMA
Missing required columns: None

MISSING VALUES
None

DUPLICATES
Exact duplicate rows:          9
Duplicate transaction records: 9

LABELS
Unique labels:  [0, 1]
Invalid labels: []

AMOUNTS
Non-numeric received: 0
Negative received:    0
Non-numeric paid:     0
Negative paid:        0

GRAPH-RELATED CHECKS
Self-transactions:    591,212
Empty source IDs:     0
Empty destination IDs: 0

TIMESTAMP
Null timestamps:      0
Out-of-order rows:    2,418,736

CLASS DISTRIBUTION
Label 0: 5,073,168
Label 1: 5,177
Laundering ratio: 0.101943%

===========================================================================
VALIDATION COMPLETE
===========================================================================

Total runtime: 23.95 seconds.
```

___

Next section is [[Phase 2.1 - Data Cleaning]]

___

Ok so, talked with claude and he suggested some improvements in this phase. 

The AMLGaurd methodology described in the handoff uses an account-level illicit label derived from transaction-level laundering labels: an account is considered illicit when it participates as the source or destination of at least one illicit transaction. The handoff also recommends preserving source-only and destination-only flags rather than throwing away that directional information.

We’ll implement exactly that, but we'll call it **`is_illicit`**, not "mule", because an account touching an illicit transaction is evidence of illicit involvement, not definitive proof that the account itself is a money mule.


# What we're creating

From:

```
Transaction table
        ↓
is_laundering = 1
        ↓
collect source + destination accounts
```

we'll create:

```
account_labels.csv

account_id
is_illicit_source
is_illicit_dest
is_illicit
```

Where:

`is_illicit_source = 1`

means the account was the **sender/source** of at least one laundering transaction.

`is_illicit_dest = 1`

means it was the **receiver/destination** of at least one laundering transaction.

`is_illicit = 1`

means either of those is true.

That lets us later study whether an account behaves more like a source/fan-out node or destination/fan-in node.

___

Output when `labels.py` is ran:

```
===========================================================================
ACCOUNT-LEVEL LABEL DERIVATION
===========================================================================

Input:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\HI-Small_Trans_clean.csv

Output:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\account_labels.csv

[1/5] Loading transaction labels...
Transactions loaded: 5,078,336

[2/5] Creating globally unique account IDs...

[3/5] Deriving source/destination illicit flags...
Laundering transactions: 5,177
Accounts involved as illicit sources: 3,376
Accounts involved as illicit destinations: 3,984

[4/5] Building account-level label table...

[5/5] Saving and verifying labels...

===========================================================================
ACCOUNT LABEL DERIVATION COMPLETE
===========================================================================

Total accounts:                 515,088
Illicit source accounts:       3,376
Illicit destination accounts:  3,984
Illicit accounts (combined):   6,357
Source only:                    2,373
Destination only:               2,981
Both source + destination:      1,003
Account illicit ratio:          1.234158%

Output saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\account_labels.csv

Runtime: 9.17 seconds
```

