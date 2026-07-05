from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", case_sensitive=True, extra="ignore")

    # Telegram Bot
    BOT_TOKEN: str = ""

    # Model paths
    VIT_MODEL_PATH: str = str(Path(__file__).parent / "models" / "best_vit_model_stage1.pth")

    # API Endpoint
    API_URL: str = "http://localhost:8000"


settings = Settings()
