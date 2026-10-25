___

## Dataset

**IBM Transactions for Anti Money Laundering**

Selected dataset:

`HI-Small_Trans.csv`

The roadmap recommends the HI-Small variant because it is the smallest variant with a relatively useful number of positive examples for development.

## Raw Dataset Fields

```
Timestamp
From Bank
Account
To Bank
Account.1
Amount Received
Receiving Currency
Amount Paid
Payment Currency
Payment Format
Is Laundering
```

The loader normalizes these columns into snake_case names.

## Important Loader Decision

Raw account numbers are not assumed to be globally unique.

Therefore we generate:

```
from_account_id = from_bank + "_" + from_account
to_account_id   = to_bank + "_" + to_account
```

This prevents accounts from different banks with the same raw account number from accidentally becoming one graph node.

## Phase 1 Results

```
Transactions:      5,078,345
Unique accounts:   515,088
Unique banks:      30,470
Laundering txns:   5,177
Laundering ratio:  0.101943%
Date range:
2022-09-01 00:00:00
→
2022-09-18 16:18:00
```

Payment formats:

```
Cheque        1,864,331
Credit Card   1,323,324
ACH             600,797
Cash            490,891
Reinvestment    481,056
Wire            171,855
Bitcoin         146,091
```

The loader's profiling function records transaction count, account count, bank count, laundering count/ratio, date range, payment formats and currency distribution.
