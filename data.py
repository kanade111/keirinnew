"""Data loading and preprocessing utilities implemented with the standard library."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils import get_logger

logger = get_logger(__name__)

MANDATORY_COLUMNS = [
    "race_id",
    "date",
    "track",
    "class",
    "grade",
    "lane_no",
    "rider_name",
    "score",
    "style",
    "backs",
    "homes",
    "starts",
    "win_rate",
    "quinella_rate",
    "top3_rate",
    "kimarite_nige",
    "kimarite_makuri",
    "kimarite_sashi",
    "kimarite_mark",
]

STYLE_MAP = {
    "逃": "escape",
    "追": "chaser",
    "両": "both",
    "自在": "flex",
}

ROLLING_COLUMNS = [
    "score",
    "backs",
    "homes",
    "starts",
    "win_rate",
    "quinella_rate",
    "top3_rate",
    "kimarite_nige",
    "kimarite_makuri",
    "kimarite_sashi",
    "kimarite_mark",
]


@dataclass
class Dataset:
    rows: List[Dict[str, object]]
    labels_top3: List[Optional[float]]
    labels_win: List[Optional[float]]


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return float(value.replace("%", "")) / 100.0
        except Exception:
            return None


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return datetime.fromisoformat(value)


def _normalize_style(row: Dict[str, object]) -> None:
    style = str(row.get("style") or "").strip()
    mapped = STYLE_MAP.get(style, style or "unknown")
    for label in {"escape", "chaser", "both", "flex", "unknown"}:
        row[f"style_{label}"] = 1.0 if mapped == label else 0.0


def load_csv(path: str | Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row: Dict[str, object] = {k: v for k, v in raw.items()}
            rows.append(row)
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def _ensure_mandatory(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    missing = [col for col in MANDATORY_COLUMNS if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing mandatory columns: {missing}")


def load_races(path: str | Path) -> List[Dict[str, object]]:
    rows = load_csv(path)
    _ensure_mandatory(rows)
    for row in rows:
        if row.get("date"):
            row["date"] = _parse_date(str(row["date"]))
        for key in ROLLING_COLUMNS + ["finish_pos", "age"]:
            if key in row:
                row[key] = _to_float(str(row[key]))
        _normalize_style(row)
    return rows


def load_cards(path: str | Path) -> List[Dict[str, object]]:
    rows = load_csv(path)
    for row in rows:
        if row.get("date"):
            row["date"] = _parse_date(str(row["date"]))
        for key in ROLLING_COLUMNS + ["age"]:
            if key in row:
                row[key] = _to_float(str(row[key]))
        _normalize_style(row)
    return rows


def _sort_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("date") or datetime.min,
            str(r.get("race_id") or ""),
            float(r.get("lane_no") or 0),
        ),
    )


def prepare_training_data(
    rows: List[Dict[str, object]],
    rolling_window: int = 5,
) -> Dataset:
    rows = _sort_rows(list(rows))
    history: Dict[str, Dict[str, List[float]]] = {}
    labels_top3: List[Optional[float]] = []
    labels_win: List[Optional[float]] = []

    for row in rows:
        rider_key = str(row.get("rider_id") or row.get("rider_name") or "")
        rider_hist = history.setdefault(rider_key, {col: [] for col in ROLLING_COLUMNS})
        for col in ROLLING_COLUMNS:
            values = rider_hist[col]
            if values:
                recent = values[-rolling_window:]
                row[f"{col}_roll_mean"] = sum(recent) / len(recent)
            else:
                row[f"{col}_roll_mean"] = None
        for col in ROLLING_COLUMNS:
            value = row.get(col)
            if isinstance(value, (int, float)):
                rider_hist[col].append(float(value))
        finish = row.get("finish_pos")
        if finish is None:
            labels_top3.append(None)
            labels_win.append(None)
        else:
            labels_top3.append(1.0 if float(finish) <= 3 else 0.0)
            labels_win.append(1.0 if float(finish) == 1 else 0.0)

    return Dataset(rows=rows, labels_top3=labels_top3, labels_win=labels_win)


def prepare_inference_data(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return _sort_rows(list(rows))

