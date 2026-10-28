___

The first 25 features were successfully generated, but I identified two definition issues:

1. `reciprocal_counterparty_count` can double-count reciprocal relationships.
2. `max_daily_transactions` can combine maxima from two different days.

So im fixing those now and adding a few **high-value temporal/network features**. ive still deliberately leaving currency-mixed amount statistics out until we establish a valid normalization method.

___


The updated feature set will add:

```
Existing
├── activity
├── counterparties
├── fan-in / fan-out
├── reciprocity
├── temporal activity
└── bank/payment diversity

New
├── cross_bank_ratio
├── reciprocal_ratio (corrected)
├── activity_burst_ratio
├── transaction_days_ratio
├── incoming/outgoing transaction ratio
├── incoming/outgoing counterparty ratio
├── sender/receiver concentration
└── net transaction flow
```

The temporal additions are justified by the Vayyala paper's discussion of bursty/transient financial behavior, and the topology features fit the AML graph framing.

___

# Desired Output:

```
===========================================================================
PHASE 4 — ACCOUNT FEATURE ENGINEERING v2
===========================================================================

[1/8] Loading transactions...
Transactions: 5,078,336

[2/8] Creating account IDs...
Accounts: 515,088

[3/8] Basic activity features...

[4/8] Counterparty and topology features...
  Calculating reciprocal relationships...

[5/8] Temporal behavior features...

[6/8] Cross-bank and diversity features...

[7/8] Counterparty concentration...

[8/8] Cleaning feature table and saving...

===========================================================================
PHASE 4 — FEATURE ENGINEERING v2 COMPLETE
===========================================================================

Accounts:  515,088
Features:  31
Runtime:   48.38 seconds

Output:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\account_features_v2.csv

Features:
  - in_transaction_count
  - out_transaction_count
  - total_transaction_count
  - in_out_transaction_ratio
  - unique_senders
  - unique_receivers
  - unique_counterparties
  - fan_in
  - fan_out
  - transaction_in_degree
  - transaction_out_degree
  - in_out_counterparty_ratio
  - reciprocal_counterparty_count
  - reciprocity_ratio
  - incoming_active_days
  - outgoing_active_days
  - active_days
  - transactions_per_active_day
  - max_daily_in_transactions
  - max_daily_out_transactions
  - max_daily_transactions
  - activity_burst_ratio
  - unique_sender_banks
  - unique_receiver_banks
  - total_counterparty_banks
  - incoming_payment_format_diversity
  - outgoing_payment_format_diversity
  - receiving_currency_diversity
  - payment_currency_diversity
  - cross_bank_ratio
  - counterparty_concentration
===========================================================================
```

___

