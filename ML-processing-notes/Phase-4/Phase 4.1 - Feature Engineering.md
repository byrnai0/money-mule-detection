___

This phase is going to produce the numerical description of every account that our GNN will eventually learn from.

The key idea is:

```
Raw transactions
        ↓
Account behavior
        +
Network structure
        +
Temporal behavior
        +
Transaction diversity
        ↓
Account feature vector
        ↓
GNN
```

___

Features I'll be building here are as follows:

## 1A. Basic account activity

For every account:

```
in_transaction_count
out_transaction_count
total_transaction_count

unique_senders
unique_receivers
unique_counterparties

unique_sender_banks
unique_receiver_banks
```

These establish basic account behavior.


## 1B. Transaction amount behavior

We'll calculate:

```
incoming_amount statistics
outgoing_amount statistics
```

such as:

```
mean
std
min
max
median
```

### But there's a catch

Our dataset contains multiple currencies.

So **we are NOT going to blindly add USD + Euro + Yuan + Bitcoin amounts together**.

That's mathematically wrong.

The Vayyala paper says transaction amount is useful, but it doesn't solve this specific IBM-AML currency issue for us.

We'll handle currency-aware amounts separately rather than create garbage features.


## 2. Transaction diversity

We'll calculate:

```
payment_format_count
payment_format_diversity
receiving_currency_diversity
payment_currency_diversity
```

For example, an account using:

```
ACH
Wire
Cash
Credit Card
```

behaves differently from one that only performs one kind of transaction.


## 3. Network topology

This is where the project starts becoming **money-mule detection** instead of generic fraud detection.

We'll calculate:

```
in_degree
out_degree

unique_in_neighbors
unique_out_neighbors

fan_in
fan_out

reciprocal_counterparty_count
reciprocity_ratio
```

Conceptually:

```
                B
                ↑
                |
A ──────────────┤
                |
                C
```

is fan-in behavior.

Whereas:

```
                B
                |
                ↓
A ─────────────→ C
                ↑
                |
                D
```

contains fan-out behavior.

The AMLGaurd and Hadinata papers explicitly identify fan-in, fan-out, cycles, scatter-gather and gather-scatter as relevant money-laundering topologies.


## 4. Temporal behavior

This is a new emphasis we are strengthening because of the Vayyala paper.

We'll calculate things such as:

```
first_transaction_time
last_transaction_time

active_days
transactions_per_active_day

max_transactions_in_one_day

average_time_between_transactions
minimum_time_between_transactions
```

And later potentially:

>burst_score

because coordinated laundering can occur in short bursts. Vayyala specifically discusses bursty and transient temporal behavior as useful anomaly signals.


## 5. Cross-bank behavior

Our graph has banks as part of the account identity.

So we can derive:

```
unique_counterparty_banks
cross_bank_transaction_count
cross_bank_ratio
```

This is especially relevant because the project ultimately wants to detect networks spanning financial institutions.

The Vayyala paper's central argument is that cross-institution relationships can reveal connections hidden in isolated institutional datasets.

___

# One very important issue: temporal leakage

This is something I want us to handle correctly from the start.

Suppose:

>September 1 → September 15

is our training period.

And an account performs suspicious activity on September 15.

We **cannot** calculate its training feature using transactions from September 16–18.

That would give the model information from the future.

So Phase 4 will have two concepts:

```
EDA features
    ↓
whole dataset
```

and eventually:

```
training features
    ↓
only transactions available up to the split time
```

That matters because Haider's work explicitly evaluates both random and chronological splits, and the Vayyala paper emphasizes temporal transaction behavior.

For our first feature-engineering run, we'll generate **full-dataset features for inspection and feature validation**. Before training, we'll make the feature generation split-aware.

___

# Desired Output:

```
===========================================================================
PHASE 4 — ACCOUNT FEATURE ENGINEERING
===========================================================================

Loading transactions...
Transactions loaded: 5,078,336

Creating account IDs...
Accounts: 515,088

[1/7] Basic transaction counts...
[2/7] Counterparty features...
[3/7] Network topology features...
[4/7] Reciprocal relationship features...
[5/7] Temporal behavior features...
[6/7] Bank and payment diversity...
[7/7] Finalizing feature table...

===========================================================================
PHASE 4 — INITIAL FEATURE ENGINEERING COMPLETE
===========================================================================

Accounts:       515,088
Features:       25
Output:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\account_features_full.csv

Runtime:        15.75 seconds

Features generated:
  - in_transaction_count
  - out_transaction_count
  - total_transaction_count
  - unique_senders
  - unique_receivers
  - unique_counterparties
  - fan_in
  - fan_out
  - in_degree
  - out_degree
  - reciprocal_counterparty_count
  - reciprocity_ratio
  - incoming_active_days
  - outgoing_active_days
  - active_days
  - transactions_per_active_day
  - max_daily_in_transactions
  - max_daily_out_transactions
  - max_daily_transactions
  - unique_sender_banks
  - unique_receiver_banks
  - incoming_payment_format_diversity
  - outgoing_payment_format_diversity
  - receiving_currency_diversity
  - payment_currency_diversity
===========================================================================
```