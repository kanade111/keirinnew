"""Utility helpers for keirin prediction pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

JST = timezone.utc
try:
    import zoneinfo  # type: ignore
except Exception:  # pragma: no cover - Python < 3.9 fallback
    zoneinfo = None  # type: ignore


def get_logger(name: str = "keirin") -> logging.Logger:
    """Return a logger configured with a sensible default format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: Optional[str | Path]) -> Dict[str, Any]:
    """Read YAML configuration, returning an empty dict if the file is missing."""
    if not path:
        return {}
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class TimeResolver:
    """Helper to resolve dates in Asia/Tokyo timezone."""

    tz_name: str = "Asia/Tokyo"

    def now(self) -> datetime:
        if zoneinfo is None:
            return datetime.now(timezone.utc)
        return datetime.now(zoneinfo.ZoneInfo(self.tz_name))

    def today_str(self) -> str:
        return self.now().strftime("%Y-%m-%d")


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def getenv_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

