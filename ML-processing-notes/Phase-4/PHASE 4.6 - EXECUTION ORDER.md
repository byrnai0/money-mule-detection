___

# Phase 4 — Feature Engineering

## 7. `src/features/account_features.py`

**Purpose:** First exploratory account feature generation. Creates basic activity, counterparty, topology, temporal, bank, payment-format, and currency-diversity features.

**When to run:** First feature-engineering pass. Mainly exploratory.

**Input:**

```text
data/processed/HI-Small_Trans_clean.csv
```

**Output:**

```text
data/processed/account_features_full.csv
```

---

## 8. `src/features/account_features_v2.py`

**Purpose:** Refines the first feature set. Fixes feature definitions and adds better temporal, reciprocal, cross-bank, and concentration features.

**When to run:** After the first exploratory feature set, when continuing Phase 4 development.

**Input:**

```text
data/processed/HI-Small_Trans_clean.csv
```

**Output:**

```text
data/processed/account_features_v2.csv
```

---

## 9. `src/features/profile_features.py`

**Purpose:** Profiles the v2 feature matrix. Measures distributions, zero percentages, class-wise means, target correlation, and highly correlated feature pairs.

**When to run:** After `account_features_v2.py`.

**Input:**

```text
data/processed/account_features_v2.csv
data/processed/account_labels.csv
```

**Output:**

```text
data/processed/feature_profile.csv
```

---

## 10. `src/features/account_features_v3.py`

**Purpose:** Final candidate feature generation for Phase 4. Adds refined topology/temporal features and currency-aware amount features. Includes the corrected reciprocal-count implementation and counterparty concentration.

**When to run:** After the v2 profiling/design stage.

**Input:**

```text
data/processed/HI-Small_Trans_clean.csv
```

**Output:**

```text
data/processed/account_features_v3.csv
```

Current verified result:

```text
515,088 accounts
38 candidate features
```

---

## 11. `src/features/profile_features_v2.py`

**Purpose:** Profiles the final candidate features and identifies redundancy, skew, weak features, and potential candidates for removal.

**When to run:** After `account_features_v3.py`.

**Input:**

```text
data/processed/account_features_v3.csv
data/processed/account_labels.csv
```

**Output:**

```text
data/processed/feature_profile_v3.csv
```

---

## 12. `src/features/finalize_features.py`

**Purpose:** Selects the approved candidate features, removes redundant/problematic features, applies exploratory `log1p` transforms to selected count-like variables, and creates the feature manifest.

**When to run:** After the final v3 profiling/feature-selection decision.

**Input:**

```text
data/processed/account_features_v3.csv
```

**Outputs:**

```text
data/processed/account_features_final.csv
data/processed/feature_manifest.csv
```

---

## 13. `src/features/verify_final_features.py`

**Purpose:** Final Phase 4 integrity check. Verifies feature structure, NaN/Inf values, target leakage, label alignment, and graph-account alignment.

**When to run:** Last step of Phase 4, immediately before Phase 5.

**Inputs:**

```text
data/processed/account_features_final.csv
data/processed/account_labels.csv
data/graphs/ibm_aml_graph.pt
```

**Current verified result:**

```text
515,088 accounts
26 final features
NaN: 0
Infinity: 0
Target leakage: NONE
Labels aligned: YES
Graph aligned: YES
```

---

# Phase 4 TL;DR

```text
account_features.py
        ↓
account_features_v2.py
        ↓
profile_features.py
        ↓
account_features_v3.py
        ↓
profile_features_v3.py
        ↓
finalize_features.py
        ↓
verify_final_features.py
```

The progression is:

```text
Generate
→ Refine
→ Profile
→ Generate final candidates
→ Profile
→ Select/transform
→ Verify
```

The final model-ready Phase 4 artifacts are:

```text
data/processed/account_features_final.csv
data/processed/account_labels.csv
data/processed/feature_manifest.csv
data/graphs/ibm_aml_graph.pt
```

---

# Important Run Rule

Do **not** run later phases before their required artifacts exist.

For Phase 5, the expected starting point is:

```text
account_features_final.csv
+
account_labels.csv
+
ibm_aml_graph.pt
```

Phase 5's exact execution order will be added once the GNN training files are implemented.
