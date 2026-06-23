"""Miner environment loading (.env): HuggingFace token, etc. (see .env.example_miners)."""

from __future__ import annotations

from core.dotenv import load_dotenv


def load_miner_env(path: str = ".env") -> bool:
    return load_dotenv(path)
