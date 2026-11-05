___

Up to this point the model could say:

```
Account X
Risk = 0.94
```

But that's not enough.

For AML, we need to answer:

> **WHY did you give this account a high risk score?**

So 8A was our first attempt at understanding the model **globally**.

We asked:

> "Across all accounts, which of our 26 features does the ensemble actually rely on?"

We used **permutation importance**.

___

### What permutation importance means

Imagine the model normally gets:

```
PR-AUC = 0.308
```

Now we take ONE feature, say:

```
outgoing_log_amount_max
```

and randomly shuffle that feature between accounts.

Everything else stays the same.

Then we run the models again.

If performance becomes:

```
PR-AUC = 0.020
```

then that feature was extremely important.

The bigger:

```
original PR-AUC - shuffled PR-AUC
```

the more important the feature was to the model's predictions.

We did this for all 26 features.

___

## Findings:

| Rank | Feature                                   |  PR-AUC drop |
| ---: | ----------------------------------------- | -----------: |
|    1 | `outgoing_log_amount_max`                 | **0.288134** |
|    2 | `log1p_incoming_payment_format_diversity` | **0.217667** |
|    3 | `log1p_receiving_currency_diversity`      | **0.118725** |
|    4 | `outgoing_log_amount_std`                 |     0.092014 |
|    5 | `log1p_out_transaction_count`             |     0.081110 |
|    6 | `log1p_unique_receiver_banks`             |     0.064940 |
|    7 | `log1p_outgoing_payment_format_diversity` |     0.057455 |

The model is heavily relying on:
```
Amount behavior
Payment-format diversity
Currency diversity
Transaction activity
Bank diversity
```

