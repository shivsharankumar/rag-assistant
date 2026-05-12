"""Application configuration from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ragdb"
    db_user: str = "postgres"
    db_password: str = "devpass"
    
    # AWS (used from Week 2)
    aws_profile: str = "rag-dev"
    aws_region: str = "us-east-1"
    
    # LangSmith (Week 2)
    langsmith_api_key: str = ""


settings = Settings()