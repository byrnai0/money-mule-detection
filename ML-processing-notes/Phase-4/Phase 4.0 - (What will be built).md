___

We'll create **three broad categories of features**.

#### 1. Transaction-level features

Things directly associated with a transaction:

```
amount_received
amount_paid
payment currency
receiving currency
payment format
timestamp
```
The IBM dataset already gives us these raw attributes.

---

#### 2. Account-level behavioral features

These are calculated by looking at all transactions involving an account.

Examples:

```
total incoming transactions
total outgoing transactions
total incoming amount
total outgoing amount
average transaction amount
transaction frequency
unique counterparties
incoming/outgoing ratio
```

These help answer:

> "How does this account normally behave?"

---

#### 3. Network / topology features

This is where we start making the project specifically about **money mule networks**.

We'll investigate features such as:

```
fan-in
fan-out
cycle participation
scatter-gather behavior
gather-scatter behavior
counterparty concentration
degree
in-degree
out-degree
```

The supporting AML graph literature explicitly identifies patterns such as fan-in, fan-out, cycles and scatter-gather as suspicious transaction topologies.

___

Here, we will create a specific feature function..

For every feature:

```
Feature
→ Definition
→ Calculation
→ Node or edge?
→ Why it matters?
→ Possible leakage?
→ Data type
```

That prevents us from randomly generating 40 features just because we can.

___

Link to other MD files of the same phase:

[[Phase 4.1 - Feature Engineering]]
[[Phase 4.2 - Feature Engineering v2]]
[[Phase 4.3 - Feature Engineering v3]]
[[Phase 4.4 - Feature Engineering v4]]
[[Phase 4.5 - Final Feature Selection]]
[[PHASE 4.6 - EXECUTION ORDER]]
