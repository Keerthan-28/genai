"""Central configuration for VeriReport AI.

Loads environment variables from `.env` (if present) and configures local Ollama
runtime defaults without requiring external API credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    """Application-wide configuration settings."""

    app_name: str = "VeriReport AI"
    app_version: str = "0.1.0"
    project_root: Path = PROJECT_ROOT

    # Directory paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    sample_data_dir: Path = PROJECT_ROOT / "data" / "samples"
    database_dir: Path = PROJECT_ROOT / "database"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"
    exports_dir: Path = PROJECT_ROOT / "outputs" / "exports"

    # Database
    database_path: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "database/verireport.db")

    # Local Ollama configuration
    llm_provider: str = "ollama"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    ollama_fallback_model: str = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    ollama_max_retries: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

    # Upload constraints
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    allowed_extensions: tuple[str, ...] = (".csv", ".xlsx", ".xls")

    # Runtime
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    """Return a fresh Settings instance reflecting current environment variables."""
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
    return Settings(
        database_path=PROJECT_ROOT / os.getenv("DATABASE_PATH", "database/verireport.db"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        ollama_fallback_model=os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b"),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        ollama_max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
