from pydantic import BaseModel


class EnrollResponse(BaseModel):
    person_id: int
    name: str


class MatchResult(BaseModel):
    bbox: list[float]
    person_id: int | None
    name: str | None
    similarity: float
