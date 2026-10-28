___

The profiler showed us that the current 31-feature set has a lot of redundancy, so we're not going to keep piling features on top of it. The next pass adds the one major missing signal: **transaction amount behavior**, plus a few better topology ratios.

created `account_features_v3.py` here.

We're going to reduce the current 31-feature set instead of blindly adding more.

### Keep / transform

```
in_transaction_count
out_transaction_count
unique_senders
unique_receivers
reciprocal_counterparty_count
reciprocity_ratio

incoming_active_days
outgoing_active_days
active_days
activity_burst_ratio

unique_sender_banks
unique_receiver_banks

incoming_payment_format_diversity
outgoing_payment_format_diversity
receiving_currency_diversity
payment_currency_diversity

counterparty_concentration
```

### Drop as redundant

```
transaction_in_degree
transaction_out_degree
fan_in              → duplicate of unique_senders
fan_out             → duplicate of unique_receivers
total_counterparty_banks / cross_bank_ratio → reconsider
max_daily_*         → don't need all versions
transactions_per_active_day → heavily redundant
total_transaction_count → may retain one overall-volume measure
unique_counterparties → potentially redundant with directional counts
```

We'll make the final decision based on the second profiling pass rather than arbitrarily deleting them.

___

After running `account_features_v3.py` we get 37 candidates.

The important thing is not the number 37. The important thing is that we now cover the major signals we wanted:

```
Account behavior
        +
Network topology
        +
Temporal behavior
        +
Bank/payment diversity
        +
Amount behavior
        ↓
37 candidate features
```

The amount layer is especially important because the earlier versions mostly measured connectivity/activity. The uploaded AML architecture paper also emphasizes transaction amount, timestamp and transaction type as core analytical attributes, while its temporal discussion supports looking at changing and bursty transaction behavior.

___

Created `profile_features_v3.py` which is the same as the previous profile_feature.py file but here,
We'll categorize the 37 features into:

### Keep

Features providing genuinely distinct information.

### Remove

Exact duplicates or extremely redundant features.

### Transform

Highly skewed variables where `log1p()` or another scaling strategy is appropriate.

### Investigate

Features that look promising but could contain methodological problems.

### Reject

Features that contribute almost nothing or are poorly defined.

# Desired Output:

```
===========================================================================
PHASE 4 — FEATURE PROFILING
===========================================================================

Loading features...
Loading account labels...
Feature rows: 515,088
Label rows:   515,088

Analyzing 37 candidate features...

===========================================================================
FEATURE PROFILE
===========================================================================
                          feature  min           max      mean   median        std  zero_pct  unique_values  illicit_mean  legit_mean  standardized_mean_difference  target_correlation
       payment_currency_diversity  0.0     10.000000  0.983469 1.000000   0.238479  3.511827             11      1.253264    0.980098                      1.145452            0.126464
     receiving_currency_diversity  0.0     10.000000  0.833539 1.000000   0.419574 18.336284             11      1.260658    0.828202                      1.030701            0.113795
              unique_sender_banks  0.0    494.000000  1.911572 1.000000   2.252679 18.336284             77      3.969325    1.885859                      0.924883            0.102112
                           fan_in  0.0    545.000000  1.971966 2.000000   2.631086 18.336284             89      4.265377    1.943308                      0.882551            0.097438
                   unique_senders  0.0    545.000000  1.971966 2.000000   2.631086 18.336284             89      4.265377    1.943308                      0.882551            0.097438
    reciprocal_counterparty_count  0.0     14.000000  0.028679 0.000000   0.262129 98.702552              7      0.223061    0.026250                      0.750821            0.082894
incoming_payment_format_diversity  0.0      6.000000  1.983180 1.000000   1.605307 18.336284              7      3.087777    1.969377                      0.696690            0.076918
                reciprocity_ratio  0.0      1.000000  0.003923 0.000000   0.037691 98.702552            121      0.029259    0.003606                      0.680617            0.075144
        max_daily_in_transactions  0.0    539.000000  2.544389 2.000000   2.996269 18.336284             89      4.490640    2.520069                      0.657675            0.072611
          incoming_log_amount_max  0.0     27.676284  8.076043 8.881870   5.098370 18.336284         385269     11.153268    8.037591                      0.611112            0.067470
outgoing_payment_format_diversity  0.0      6.000000  1.800355 1.000000   1.344022  3.511827              7      2.607991    1.790262                      0.608419            0.067173
            unique_receiver_banks  0.0    618.000000  1.853060 1.000000   2.018368  3.511827             42      3.033034    1.838315                      0.591923            0.065351
      incoming_currency_abs_z_max  0.0     10.619470  1.211445 1.350942   0.921454 18.336284         298430      1.715820    1.205143                      0.554208            0.061187
             activity_burst_ratio  1.0      7.528090  1.484803 1.358025   0.560315  0.000000           3010      1.790483    1.480983                      0.552368            0.060984
          incoming_log_amount_std  0.0     16.528176  1.384123 1.027025   1.665940 41.566101         300975      2.245348    1.373362                      0.523420            0.057788
          outgoing_log_amount_max  0.0     27.676284  8.773571 8.780761   4.041329  3.511827         418080     10.816278    8.748046                      0.511770            0.056502
             incoming_active_days  0.0     17.000000  3.688119 1.000000   3.719661 18.336284             17      5.544439    3.664923                      0.505292            0.055787
                      active_days  1.0     17.000000  4.815265 3.000000   3.641947  0.000000             16      6.617587    4.792743                      0.501063            0.055320
             in_transaction_count  0.0   1084.000000  9.859162 2.000000  12.973989 18.336284            110     16.073934    9.781503                      0.485004            0.053547
      outgoing_currency_abs_z_max  0.0     10.553313  1.263100 1.340724   0.826606  3.511827         360772      1.613167    1.258725                      0.428791            0.047341
         incoming_log_amount_mean  0.0     22.931802  6.052564 6.491602   3.929845 18.336284         404008      7.707741    6.031881                      0.426444            0.047082
          outgoing_log_amount_std  0.0     13.550967  1.522926 0.607904   1.965140 45.890605         278690      2.291485    1.513322                      0.395983            0.043719
            unique_counterparties  1.0  14775.000000  3.943932 3.000000  25.245032  0.000000            103     13.159037    3.828782                      0.369588            0.040804
             outgoing_active_days  0.0     15.000000  3.010592 2.000000   3.170022  3.511827             16      4.090294    2.997101                      0.344854            0.038074
           max_daily_transactions  1.0  26904.000000  4.881234 4.000000  45.722974  0.000000            100     18.853233    4.706643                      0.309398            0.034159
          total_transaction_count  1.0 169756.000000 19.718324 6.000000 287.508948  0.000000            302    102.751455   18.680759                      0.292411            0.032284
      transactions_per_active_day  1.0  16975.600000  3.110316 2.125000  28.739845  0.000000            823     11.385197    3.006915                      0.291521            0.032185
                          fan_out  0.0  14230.000000  1.971966 1.000000  24.197103  3.511827             51      8.893661    1.885474                      0.289629            0.031977
                 unique_receivers  0.0  14230.000000  1.971966 1.000000  24.197103  3.511827             51      8.893661    1.885474                      0.289629            0.031977
       max_daily_out_transactions  0.0  26365.000000  2.519894 1.000000  44.697793  3.511827             59     14.809029    2.366331                      0.278374            0.030734
            out_transaction_count  0.0 168672.000000  9.859162 2.000000 285.146466  3.511827            265     86.677521    8.899255                      0.272766            0.030115
     incoming_currency_abs_z_mean  0.0      9.363196  0.702641 0.633762   0.594034 18.336284         411196      0.854125    0.700748                      0.258195            0.028506
         outgoing_log_amount_mean  0.0     25.554461  7.067983 6.960116   3.216164  3.511827         444877      7.808599    7.058728                      0.233157            0.025742
                fan_out_intensity  0.0      1.000000  0.607740 0.500000   0.353698  3.511827           1580      0.532325    0.608683                     -0.215883           -0.023835
         in_out_transaction_ratio  0.0     90.000000  3.530046 1.000000   7.297562 21.848111           5031      4.285673    3.520603                      0.104839            0.011575
     outgoing_currency_abs_z_mean  0.0      8.349445  0.847125 0.777161   0.550440  3.511827         471008      0.877954    0.846739                      0.056709            0.006261
                 fan_in_intensity  0.0      1.000000  0.424828 0.250000   0.389461 18.336284            759      0.438918    0.424652                      0.036631            0.004044

===========================================================================
HIGHLY CORRELATED FEATURE PAIRS
===========================================================================
unique_senders                           fan_in                                   1.0000
unique_receivers                         fan_out                                  1.0000
out_transaction_count                    max_daily_out_transactions               0.9997
transactions_per_active_day              max_daily_transactions                   0.9992
total_transaction_count                  transactions_per_active_day              0.9991
out_transaction_count                    total_transaction_count                  0.9990
total_transaction_count                  max_daily_transactions                   0.9989
total_transaction_count                  max_daily_out_transactions               0.9988
transactions_per_active_day              max_daily_out_transactions               0.9986
out_transaction_count                    transactions_per_active_day              0.9986
unique_receivers                         max_daily_out_transactions               0.9986
fan_out                                  max_daily_out_transactions               0.9986
out_transaction_count                    unique_receivers                         0.9985
out_transaction_count                    fan_out                                  0.9985
max_daily_out_transactions               max_daily_transactions                   0.9982
out_transaction_count                    max_daily_transactions                   0.9978
unique_receivers                         total_transaction_count                  0.9971
total_transaction_count                  fan_out                                  0.9971
unique_counterparties                    max_daily_transactions                   0.9970
unique_receivers                         transactions_per_active_day              0.9966
fan_out                                  transactions_per_active_day              0.9966
unique_receivers                         max_daily_transactions                   0.9962
fan_out                                  max_daily_transactions                   0.9962
unique_counterparties                    transactions_per_active_day              0.9956
unique_counterparties                    fan_out                                  0.9952
unique_receivers                         unique_counterparties                    0.9952
total_transaction_count                  unique_counterparties                    0.9949
unique_counterparties                    max_daily_out_transactions               0.9944
out_transaction_count                    unique_counterparties                    0.9941
unique_senders                           unique_sender_banks                      0.9623
fan_in                                   unique_sender_banks                      0.9623

===========================================================================
Profile saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\feature_profile_v3.csv
===========================================================================
```

