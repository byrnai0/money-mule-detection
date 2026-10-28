___

# What the v3 profile says

## Strong candidates

```
payment_currency_diversity
receiving_currency_diversity
unique_sender_banks
fan_in / unique_senders
reciprocal_counterparty_count
incoming_payment_format_diversity
reciprocity_ratio
max_daily_in_transactions
incoming_log_amount_max
incoming_currency_abs_z_max
outgoing_payment_format_diversity
unique_receiver_banks
activity_burst_ratio
incoming_log_amount_std
outgoing_log_amount_max
incoming_active_days
active_days
```

For example, `payment_currency_diversity` has the strongest simple association in the current set, with correlation ≈ **0.126**, and illicit accounts average 1.253 currencies versus 0.980 for legitimate accounts.

`unique_sender_banks` is also useful:

Illicit:     3.969

Legitimate:  1.886

with correlation ≈ **0.102**.

And `incoming_log_amount_max` has a meaningful difference:

Illicit:     11.15

Legitimate:   8.04

which is interesting because we're finally seeing amount behavior contribute beyond simple transaction counts.

---

# But we have a huge redundancy problem

The profiler confirms that many features are essentially measuring the same thing.

For example:

```
out_transaction_count
max_daily_out_transactions
total_transaction_count
transactions_per_active_day
unique_receivers
fan_out
unique_counterparties
```

have correlations around **0.99+** with each other.

Keeping all of them would give the model a bunch of slightly different measurements of account activity.

We don't need that.

---

# The final feature philosophy

We want each feature to answer a **different question**.

### 1. How active is the account?

Keep:

>log_total_transaction_count

One activity measure is enough.

### 2. How many different entities interact with it?

Keep:

```
log_unique_senders
log_unique_receivers
```

These directly represent incoming/outgoing network breadth.

### 3. Does it behave like an aggregator or distributor?

Keep:

```
fan_in
fan_out
```

but because they're identical to the unique counterpart counts, we don't need both.

We'll use:

```
unique_senders
unique_receivers
```

as the actual features and describe their interpretation as fan-in/fan-out.

### 4. Is activity unusually bursty?

Keep:

>activity_burst_ratio

### 5. Is behavior persistent over time?

Keep:

```
active_days
incoming_active_days
outgoing_active_days
```

Maybe we'll eventually drop one if correlation analysis says they're redundant, but they represent different directions.

### 6. Does it interact with multiple financial institutions?

Keep:

```
unique_sender_banks
unique_receiver_banks
```

### 7. Is transaction behavior diverse?

Keep:

```
payment_currency_diversity
receiving_currency_diversity
incoming_payment_format_diversity
outgoing_payment_format_diversity
```

### 8. Does it have reciprocal relationships?

Keep:

```
reciprocal_counterparty_count
reciprocity_ratio
```
after fixing the count bug.

### 9. Is its activity concentrated around a few counterparties?

Keep:

>counterparty_concentration

### 10. Does it handle unusual transaction amounts?

Keep:

```
incoming_log_amount_mean
incoming_log_amount_std
incoming_log_amount_max

outgoing_log_amount_mean
outgoing_log_amount_std
outgoing_log_amount_max

incoming_currency_abs_z_mean
incoming_currency_abs_z_max

outgoing_currency_abs_z_mean
outgoing_currency_abs_z_max
```

But we'll let the next correlation pass remove redundant ones.

___

After removing obvious duplicates, we're roughly here:

```
ACTIVITY
1.  log_total_transaction_count

NETWORK
2.  log_unique_senders
3.  log_unique_receivers
4.  counterparty_concentration
5.  reciprocal_counterparty_count
6.  reciprocity_ratio

TEMPORAL
7.  active_days
8.  incoming_active_days
9.  outgoing_active_days
10. activity_burst_ratio

BANK
11. unique_sender_banks
12. unique_receiver_banks

DIVERSITY
13. payment_currency_diversity
14. receiving_currency_diversity
15. incoming_payment_format_diversity
16. outgoing_payment_format_diversity

AMOUNT
17. incoming_log_amount_mean
18. incoming_log_amount_std
19. incoming_log_amount_max
20. outgoing_log_amount_mean
21. outgoing_log_amount_std
22. outgoing_log_amount_max
23. incoming_currency_abs_z_mean
24. incoming_currency_abs_z_max
25. outgoing_currency_abs_z_max
```
Approximately **25 useful candidates**.

___

Phase 4 is officially **complete**. The final verification passed with:

```
515,088 accounts
26 final features
NaN: 0
Infinity: 0
Target leakage: NONE
Labels aligned: YES
Graph aligned: YES
```