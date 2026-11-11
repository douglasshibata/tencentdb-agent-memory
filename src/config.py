import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    TENCENTDB_HOST: str = "10.0.0.1"
    TENCENTDB_PORT: int = 80
    TENCENTDB_USERNAME: str = "root"
    TENCENTDB_KEY: str = "your_secret_api_key_here"
    TENCENTDB_DATABASE_NAME: str = "agent_memory_db"
    TENCENTDB_COLLECTION_NAME: str = "memories"
    TENCENTDB_REGION: str = "ap-guangzhou"

    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_MODEL: str = "mock-embedding-v1"
    OPENAI_API_KEY: Optional[str] = None
    HUNYUAN_SECRET_ID: Optional[str] = None
    HUNYUAN_SECRET_KEY: Optional[str] = None

    MEMORY_DEDUPLICATION_THRESHOLD: float = 0.85
    MEMORY_DECAY_RATE: float = 0.05
    MEMORY_SHORT_TERM_LIMIT: int = 10

    MOCK_MODE: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Load settings singleton
settings = Settings()
