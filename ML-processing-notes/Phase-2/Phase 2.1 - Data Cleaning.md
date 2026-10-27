___

So now the data has been validated, and we got 9 duplicate transactions, twice.. so 18 rows are duplicate. 

We will delete 9 of them to make the duplicate data appear once.

Current status:
Phase 1 — Dataset ingestion       ✅ DONE
Phase 2 — Validation              ✅ DONE
Phase 2 — Cleaning                ⏳ NEXT
Phase 3 — Graph construction      ⏸ WAITING

Because each of those 9 records appears twice, we'll remove **one copy of each**, leaving:

5,078,345 → 5,078,336 transactions

Ok, so I created another `cleaning.py` file under the preprocessing folder called `cleaning-v2.py` so that they can be distinguished easily

cleaning.py
    ↓
"What's wrong with the dataset?"

cleaning-v2.py
    ↓
"Apply the cleaning decisions"

___

Ok, when i ran the v2 cleaning file, I did not get the output. When checked with GPT, he said to add `print("CLEANING V2 STARTED", flush=True)` at the first line of this code, I did and then got the output. 

What was happening is most likely **output buffering**.

Python can buffer `stdout`, especially when a script is being run through VS Code, PowerShell, or another terminal environment. Your program was executing, but some of the progress text wasn't being displayed immediately

the `flush=True` explicitly told Python:

> "Send this text to the terminal right now. Don't keep it in the output buffer."

___

So phase 2 is completed

```
Raw:
5,078,345 transactions

Exact duplicates:
9 records

Processed:
5,078,336 transactions

Laundering:
5,177 before
5,177 after

Remaining duplicates:
0
```

The processed file now lives at:

>data/processed/HI-Small_Trans_clean.csv
