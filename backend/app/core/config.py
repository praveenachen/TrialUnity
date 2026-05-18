from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TrialUnity"
    app_env: str = "development"
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    enable_llm: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
