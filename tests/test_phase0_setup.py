"""Phase 0 setup and configuration verification tests (Ollama configuration)."""

from __future__ import annotations

import importlib

from backend.llm_client import OllamaClientWrapper
from config.settings import PROJECT_ROOT, Settings, get_settings


def test_project_directory_structure_exists() -> None:
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/samples",
        "backend",
        "frontend",
        "frontend/pages",
        "frontend/components",
        "database",
        "outputs/reports",
        "outputs/logs",
        "outputs/exports",
        "tests",
        "config",
    ]
    for rel in expected_dirs:
        path = PROJECT_ROOT / rel
        assert path.exists() and path.is_dir(), f"Missing directory: {rel}"


def test_required_root_files_exist() -> None:
    for filename in ["requirements.txt", ".env.example", ".gitignore"]:
        assert (PROJECT_ROOT / filename).is_file(), f"Missing file: {filename}"


def test_gitignore_protects_env_and_secrets() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".venv/" in gitignore


def test_approved_packages_importable() -> None:
    required_modules = [
        "pandas",
        "numpy",
        "requests",
        "pydantic",
        "pymupdf",  # PyMuPDF
        "docx",  # python-docx
        "openpyxl",
        "streamlit",
        "plotly",
        "pytest",
        "dotenv",
        "sqlite3",
    ]
    for mod in required_modules:
        importlib.import_module(mod)


def test_requirements_reflect_ollama_override() -> None:
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [
        line.strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(packages)
    assert "requests" in joined
    assert "google-generativeai" not in joined
    assert "google-genai" not in joined
    assert "openai" not in joined
    assert "anthropic" not in joined


def test_ollama_client_handles_unreachable_service_gracefully() -> None:
    settings = Settings(ollama_base_url="http://127.0.0.1:59999", ollama_max_retries=1)
    wrapper = OllamaClientWrapper(settings=settings)
    check = wrapper.verify_model_availability()
    assert check.provider == "ollama"
    assert check.service_reachable is False
    assert check.model_verified is False
    assert "unreachable" in check.message.lower()


def test_ollama_model_verification_with_mock_models(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:latest")
    monkeypatch.setenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")

    settings = get_settings()
    wrapper = OllamaClientWrapper(settings=settings)

    monkeypatch.setattr(
        wrapper,
        "list_available_models",
        lambda: ["nomic-embed-text:latest", "llama3.2:3b", "llama3.2:latest"],
    )
    check = wrapper.verify_model_availability()
    assert check.service_reachable is True
    assert check.model_verified is True
    assert check.selected_model == "llama3.2:latest"


def test_ollama_model_fallback_skips_embedding_models(monkeypatch) -> None:
    settings = Settings(ollama_model="missing:latest", ollama_fallback_model="also-missing:latest")
    wrapper = OllamaClientWrapper(settings=settings)

    monkeypatch.setattr(
        wrapper,
        "list_available_models",
        lambda: ["nomic-embed-text:latest", "all-minilm:latest", "llama3.2:3b"],
    )
    check = wrapper.verify_model_availability()
    assert check.model_verified is True
    assert check.selected_model == "llama3.2:3b"
