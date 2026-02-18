############# PYDANTIC SETTINGS ###############
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Project metadata
    PROJECT_NAME: str = "SMS Spam Shield"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Model paths
    MODEL_REGISTRY_PATH: str = "./models"
    DATASET_PATH: str = "./dataset/processed/processed_dataset_balanced.csv"
    FEEDBACK_STORE_PATH: str = "./data/feedback.json"
    
    # ML settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Frontend
    FRONTEND_URL: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
