"""Minimal .env loader (no third-party dependency).

Reads ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` without overriding
values that are already set. Supports ``export KEY=VALUE`` and quoted values.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> bool:
    file_path = Path(path)
    if not file_path.is_file():
        return False
    for raw in file_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return True
