___

We now have:

```
515,088 accounts
31 candidate features
5,078,336 transactions
Runtime: 48.38 s
```

The feature pipeline is also running fast enough that we can afford several iterations without the project becoming painful.

## What this means

Our current feature set now covers:

```
Behavior
├── transaction volume
├── in/out behavior
└── counterparty behavior

Topology
├── fan-in
├── fan-out
├── reciprocal relationships
└── concentration

Temporal
├── active days
├── transaction frequency
├── daily peaks
└── burst behavior

Institutional / categorical diversity
├── banks
├── payment formats
└── currencies
```

Now we gotta check which of these features are useful.

We're going to inspect every feature for:

```
1. Distribution
2. Zero percentage
3. Missing/invalid values
4. Scale
5. Redundancy/correlation
6. Relationship with is_illicit
7. Potential leakage
```

This is important because we already have several features that are mathematically related.

For example:

```
in_transaction_count
transaction_in_degree
```

are currently essentially the same information.

Likewise:

```
unique_senders
fan_in
```

are identical.

And:

```
unique_receivers
fan_out
```

are identical.

We **don't necessarily need all of those in the final model**.

___

We're **not trying to maximize the number of features**.

The final objective is more like:

```
31 candidate features
        ↓
remove duplicates/redundancy
        ↓
add genuinely missing AML signals
        ↓
remove leakage
        ↓
transform heavily-skewed variables
        ↓
normalize
        ↓
final ~15–30 meaningful features
```

A smaller, defensible feature set is better than a 100-column  model.

___

## What the profiler tells us

### 1. The strongest individual signals

By standardized mean difference / target correlation, the most interesting candidates are:

```
payment_currency_diversity
receiving_currency_diversity
total_counterparty_banks
unique_sender_banks
unique_senders / fan_in
reciprocal_counterparty_count
incoming_payment_format_diversity
reciprocity_ratio
counterparty_concentration
max_daily_in_transactions
activity_burst_ratio
```

For example, illicit accounts have substantially higher average payment-currency diversity:

```
Illicit: 1.253
Legitimate: 0.980
```

and receiving-currency diversity:

```
Illicit: 1.261
Legitimate: 0.828
```

Those are much more interesting than simply saying "this account made lots of transactions."

The reciprocal relationship features are also interesting despite being sparse: only about **1.30% of accounts have any reciprocal counterparty**, but the illicit/legitimate difference is relatively strong among those accounts.

___

# Desired Output:

```
===========================================================================
PHASE 4 — FEATURE PROFILING
===========================================================================

Loading features...
Loading account labels...
Feature rows: 515,088
Label rows:   515,088

Analyzing 31 candidate features...

===========================================================================
FEATURE PROFILE
===========================================================================
                          feature      min          max      mean   median        std  zero_pct  unique_values  illicit_mean  legit_mean  standardized_mean_difference  target_correlation
       payment_currency_diversity 0.000000     10.00000  0.983469 1.000000   0.238479  3.511827             11      1.253264    0.980098                      1.145452            0.126464
     receiving_currency_diversity 0.000000     10.00000  0.833539 1.000000   0.419574 18.336284             11      1.260658    0.828202                      1.030701            0.113795
         total_counterparty_banks 1.000000   1112.00000  3.764632 3.000000   3.416102  0.000000             87      7.002360    3.724174                      0.959627            0.105948
              unique_sender_banks 0.000000    494.00000  1.911572 1.000000   2.252679 18.336284             77      3.969325    1.885859                      0.924883            0.102112
                   unique_senders 0.000000    545.00000  1.971966 2.000000   2.631086 18.336284             89      4.265377    1.943308                      0.882551            0.097438
                           fan_in 0.000000    545.00000  1.971966 2.000000   2.631086 18.336284             89      4.265377    1.943308                      0.882551            0.097438
    reciprocal_counterparty_count 0.000000      7.00000  0.014339 0.000000   0.131064 98.702552              7      0.111531    0.013125                      0.750821            0.082894
incoming_payment_format_diversity 0.000000      6.00000  1.983180 1.000000   1.605307 18.336284              7      3.087777    1.969377                      0.696690            0.076918
                reciprocity_ratio 0.000000      0.50000  0.001961 0.000000   0.018846 98.702552            121      0.014630    0.001803                      0.680617            0.075144
       counterparty_concentration 0.000105      1.00000  0.522756 0.500000   0.264601  0.000000          29861      0.347034    0.524952                     -0.672400           -0.074236
        max_daily_in_transactions 0.000000    539.00000  2.544389 2.000000   2.996269 18.336284             89      4.490640    2.520069                      0.657675            0.072611
outgoing_payment_format_diversity 0.000000      6.00000  1.800355 1.000000   1.344022  3.511827              7      2.607991    1.790262                      0.608419            0.067173
            unique_receiver_banks 0.000000    618.00000  1.853060 1.000000   2.018368  3.511827             42      3.033034    1.838315                      0.591923            0.065351
             activity_burst_ratio 1.000000      7.52809  1.484803 1.358025   0.560315  0.000000           3010      1.790483    1.480983                      0.552368            0.060984
             incoming_active_days 0.000000     17.00000  3.688119 1.000000   3.719661 18.336284             17      5.544439    3.664923                      0.505292            0.055787
                      active_days 1.000000     17.00000  4.815265 3.000000   3.641947  0.000000             16      6.617587    4.792743                      0.501063            0.055320
             in_transaction_count 0.000000   1084.00000  9.859162 2.000000  12.973989 18.336284            110     16.073934    9.781503                      0.485004            0.053547
            transaction_in_degree 0.000000   1084.00000  9.859162 2.000000  12.973989 18.336284            110     16.073934    9.781503                      0.485004            0.053547
        in_out_counterparty_ratio 0.000000     90.00000  1.258797 1.000000   1.944345 21.848111            323      2.119880    1.248037                      0.448399            0.049506
            unique_counterparties 1.000000  14775.00000  3.943932 3.000000  25.245032  0.000000            103     13.159037    3.828782                      0.369588            0.040804
             outgoing_active_days 0.000000     15.00000  3.010592 2.000000   3.170022  3.511827             16      4.090294    2.997101                      0.344854            0.038074
                 cross_bank_ratio 0.054264      1.00000  0.987731 1.000000   0.052527  0.000000            468      0.971076    0.987939                     -0.321026           -0.035443
           max_daily_transactions 1.000000  26904.00000  4.881234 4.000000  45.722974  0.000000            100     18.853233    4.706643                      0.309398            0.034159
          total_transaction_count 1.000000 169756.00000 19.718324 6.000000 287.508948  0.000000            302    102.751455   18.680759                      0.292411            0.032284
      transactions_per_active_day 1.000000  16975.60000  3.110316 2.125000  28.739845  0.000000            823     11.385197    3.006915                      0.291521            0.032185
                 unique_receivers 0.000000  14230.00000  1.971966 1.000000  24.197103  3.511827             51      8.893661    1.885474                      0.289629            0.031977
                          fan_out 0.000000  14230.00000  1.971966 1.000000  24.197103  3.511827             51      8.893661    1.885474                      0.289629            0.031977
       max_daily_out_transactions 0.000000  26365.00000  2.519894 1.000000  44.697793  3.511827             59     14.809029    2.366331                      0.278374            0.030734
            out_transaction_count 0.000000 168672.00000  9.859162 2.000000 285.146466  3.511827            265     86.677521    8.899255                      0.272766            0.030115
           transaction_out_degree 0.000000 168672.00000  9.859162 2.000000 285.146466  3.511827            265     86.677521    8.899255                      0.272766            0.030115
         in_out_transaction_ratio 0.000000     90.00000  3.530046 1.000000   7.297562 21.848111           5031      4.285673    3.520603                      0.104839            0.011575

===========================================================================
HIGHLY CORRELATED FEATURE PAIRS
===========================================================================
in_transaction_count                     transaction_in_degree                    1.0000
out_transaction_count                    transaction_out_degree                   1.0000
unique_senders                           fan_in                                   1.0000
unique_receivers                         fan_out                                  1.0000
out_transaction_count                    max_daily_out_transactions               0.9997
transaction_out_degree                   max_daily_out_transactions               0.9997
transactions_per_active_day              max_daily_transactions                   0.9992
total_transaction_count                  transactions_per_active_day              0.9991
out_transaction_count                    total_transaction_count                  0.9990
total_transaction_count                  transaction_out_degree                   0.9990
total_transaction_count                  max_daily_transactions                   0.9989
total_transaction_count                  max_daily_out_transactions               0.9988
transactions_per_active_day              max_daily_out_transactions               0.9986
out_transaction_count                    transactions_per_active_day              0.9986
transaction_out_degree                   transactions_per_active_day              0.9986
unique_receivers                         max_daily_out_transactions               0.9986
fan_out                                  max_daily_out_transactions               0.9986
out_transaction_count                    unique_receivers                         0.9985
out_transaction_count                    fan_out                                  0.9985
unique_receivers                         transaction_out_degree                   0.9985
fan_out                                  transaction_out_degree                   0.9985
max_daily_out_transactions               max_daily_transactions                   0.9982
out_transaction_count                    max_daily_transactions                   0.9978
transaction_out_degree                   max_daily_transactions                   0.9978
total_transaction_count                  unique_receivers                         0.9971
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
unique_counterparties                    transaction_out_degree                   0.9941
unique_senders                           unique_sender_banks                      0.9623
fan_in                                   unique_sender_banks                      0.9623

===========================================================================
Profile saved to:
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\data\processed\feature_profile.csv
```

