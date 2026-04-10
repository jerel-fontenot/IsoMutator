"""
ALGORITHM SUMMARY:
isomutator Configuration Manager (src/isomutator/core/config.py)
Loads variables from .env and ensures strict typing for remote AI targeting.

TECHNOLOGY QUIRKS:
- Pydantic V2 URLs: We cast the validated `AnyHttpUrl` object back to a standard string 
  during the validation phase. This allows the rest of the codebase to treat the URL 
  as a pure string (preventing type comparison errors) while maintaining strict schema checks.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, TypeAdapter, field_validator

_url_validator = TypeAdapter(AnyHttpUrl)

class IsoConfig(BaseSettings):
    batch_size: int = 4
    max_wait_seconds: float = 1.0
    shutdown_timeout: float = 15.0
    db_path: str = "data/isomutator.db"
    worker_count: int = 0

    # Remote Targeting Defaults
    target_url: str = "http://192.9.159.125:8000"
    attacker_url: str = "http://192.9.159.125:11434"

    @field_validator('target_url', 'attacker_url')
    @classmethod
    def validate_and_strip_url(cls, v: str) -> str:
        """Validates the URL schema and strips trailing slashes to prevent route doubling."""
        valid_url = _url_validator.validate_python(v)
        return str(valid_url).rstrip("/")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        env_prefix="ISO_",
        extra="ignore"
    )

# Instantiate a global singleton
settings = IsoConfig()