import csv
import math
import random
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from schemas.analysis import LiveDetectionResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "graphs" / "ensemble_predictions.csv"
ENSEMBLE_WEIGHTS = {"GraphSAGE": 0.78, "GATv2": 0.19, "GIN": 0.03}
THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_test_predictions() -> tuple[dict, ...]:
    if not PREDICTIONS_FILE.is_file():
        raise FileNotFoundError(PREDICTIONS_FILE)

    model_columns = {
        "GraphSAGE": "sage_test_prob",
        "GATv2": "gatv2_test_prob",
        "GIN": "gin_test_prob",
    }
    rows = []

    with PREDICTIONS_FILE.open("r", newline="", encoding="utf-8-sig") as predictions_file:
        for row in csv.DictReader(predictions_file):
            try:
                probabilities = {
                    model: float(row[column])
                    for model, column in model_columns.items()
                }
                test_label = int(row["is_illicit"])
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue

            if (
                test_label not in (0, 1)
                or not row.get("account_id")
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())
            ):
                continue

            rows.append({
                "account_id": row["account_id"],
                "node_id": node_id,
                "test_label": test_label,
                "model_probabilities": probabilities,
            })

    if not rows:
        raise ValueError("No valid held-out predictions are available.")

    return tuple(rows)


@router.get("/live-random-account", response_model=LiveDetectionResponse)
def live_random_account() -> LiveDetectionResponse:
    try:
        row = random.choice(load_test_predictions())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Saved held-out model predictions are unavailable.",
        ) from exc

    score = sum(
        row["model_probabilities"][name] * ENSEMBLE_WEIGHTS[name]
        for name in ENSEMBLE_WEIGHTS
    )
    prediction = int(score >= THRESHOLD)
    return LiveDetectionResponse(
        account_id=row["account_id"],
        node_id=row["node_id"],
        score=score,
        threshold=THRESHOLD,
        predicted_label="illicit" if prediction else "legitimate",
        test_label="illicit" if row["test_label"] else "legitimate",
        model_probabilities=row["model_probabilities"],
        source="Random account from saved held-out predictions; ensemble score recomputed from saved model probabilities for this request.",
        scored_at=datetime.now(timezone.utc).isoformat(),
    )