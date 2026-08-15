Team Findings / Progress Update

[!note] Current Update

What I completed

I finished the initial dataset loading and validation for the project.

We are using the IBM AML HI-Small_Trans.csv dataset because our project is focused on detecting money-mule networks, where accounts need to become graph nodes and transactions become edges.

The dataset contains around 5 million transactions, 515k accounts and 30k banks. Only 5,177 transactions are labelled as laundering, which means the dataset is heavily imbalanced.

I also verified the dataset before doing any graph work:

no missing values
no invalid labels
no negative amounts
no invalid account IDs
timestamps are present
there are only 9 actual exact duplicate transaction records

Those 9 duplicate records will be removed from the processed copy, while the original dataset stays untouched.

There are also around 591k transactions where the source and destination account are the same. We are NOT deleting these yet because we haven't established whether they should be considered invalid or simply a characteristic of the dataset.

Current status
Phase 1 ✅ Dataset ingestion
Phase 2 ✅ Data validation
Phase 2 → Cleaning duplicate records
Phase 3 → Graph construction
Next

Once the cleaned dataset is ready, we'll start building the actual account transaction graph:

Account = Node
Transaction = Directed Edge

That is where the actual money-mule network analysis begins.

Quick Current Status
[✓] Project research collected
[✓] Dataset selected
[✓] Loader implemented
[✓] Dataset loaded
[✓] Dataset profiled
[✓] Data validated
[→] Remove exact duplicates
[ ] Build account graph
[ ] Engineer graph features
[ ] Implement GNN
[ ] Train / evaluate
[ ] Explainability
[ ] Quantum-inspired optimization
[ ] Backend API