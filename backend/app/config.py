from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://bosani:1234567890@localhost:5432/latihan_face"
    face_match_threshold: float = 0.5
    liveness_threshold: float = 0.7
    insightface_provider: str = "auto"
    insightface_det_size: int = 320

    class Config:
        env_file = ".env"


settings = Settings()
