from pydantic import BaseModel


class LiveDetectionResponse(BaseModel):
    account_id: str
    node_id: int
    score: float
    threshold: float
    predicted_label: str
    test_label: str
    model_probabilities: dict[str, float]
    source: str
    scored_at: str