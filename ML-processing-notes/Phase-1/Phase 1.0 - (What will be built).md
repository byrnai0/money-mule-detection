___

### Phase 1 — Dataset Ingestion ✅

We:

- selected IBM `HI-Small_Trans.csv`
- created the loader
- normalized the schema
- created globally unique account IDs
- profiled the dataset

Result:

```
5,078,345 transactions
515,088 accounts
30,470 banks
5,177 laundering transactions
```

The loader deliberately creates `bank_account` composite IDs because raw account numbers are only unique within a bank.

___

Link to other MD files of the same phase:
[[Phase 1.1 — Dataset Ingestion]]