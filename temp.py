import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from ingestion.loader import load_transactions


CSV_PATH = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"

print("Loading dataset...")
df = load_transactions(CSV_PATH)

print("Finding exact duplicate transactions...")

dupes = df[
    df.duplicated(keep=False)
].sort_values(
    by=[
        "timestamp",
        "from_account_id",
        "to_account_id",
    ]
)

print("\nDuplicate records:\n")
print(dupes.to_string(index=False))

print(f"\nTotal duplicate rows shown: {len(dupes)}")