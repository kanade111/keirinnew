"""Betting strategy helpers based on plain Python data structures."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils import dump_json

DEFAULT_THRESHOLDS = {
    "gachi": {"a_rate": 0.85},
    "blue": {"a_rate": 0.75, "ct": 0.55},
    "twilight": {"a_rate_min": 0.55},
    "red": {"ct": 0.35},
}


@dataclass
class BetSuggestion:
    bet_type: str
    pattern: str
    probability: float


def classify_zone(a_rate: float, ct_value: float, thresholds: Dict[str, Any]) -> str:
    if a_rate >= thresholds.get("gachi", {}).get("a_rate", 0.85):
        return "gachi"
    if (
        a_rate >= thresholds.get("blue", {}).get("a_rate", 0.75)
        or ct_value >= thresholds.get("blue", {}).get("ct", 0.55)
    ):
        return "blue"
    if a_rate >= thresholds.get("twilight", {}).get("a_rate_min", 0.55):
        return "twilight"
    return "red"


def _group_by_race(predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in predictions:
        rid = str(row.get("race_id"))
        grouped.setdefault(rid, []).append(row)
    return grouped


def _zone_templates(zone: str, abc: List[str], rates: List[float], extras: List[str], ct_value: float) -> List[BetSuggestion]:
    a, b, c = abc
    templates: List[BetSuggestion] = []
    if zone == "gachi":
        templates.append(BetSuggestion("trifecta_box", f"{a}={b}={c}", ct_value))
    elif zone == "blue":
        d = extras[0] if extras else c
        templates.extend(
            [
                BetSuggestion("trifecta_box", f"{a}={b}={c}", ct_value),
                BetSuggestion("trifecta_box", f"{a}={b}={d}", max(min(rates[0], rates[1]), 0.1)),
                BetSuggestion("trifecta", f"{a}->{b}->{c}", rates[0] * rates[1] * 0.5),
            ]
        )
    elif zone == "twilight":
        d = extras[0] if extras else c
        templates.extend(
            [
                BetSuggestion("trifecta_box", f"{a}={b}={c}", ct_value * 0.8),
                BetSuggestion("trifecta_box", f"{a}={b}={d}", max(rates[0] * rates[1], 0.05)),
                BetSuggestion("trifecta_box", f"{a}={c}={d}", max(rates[0] * rates[2], 0.05)),
                BetSuggestion("trifecta", f"{a}->{b}->{d}", rates[0] * rates[1] * 0.4),
            ]
        )
    else:
        d = extras[0] if extras else c
        e = extras[1] if len(extras) > 1 else d
        templates.extend(
            [
                BetSuggestion("trifecta_box", f"{a}={b}={c}", max(ct_value, 0.1)),
                BetSuggestion("trifecta_box", f"{a}={b}={d}", max(rates[0] * rates[1], 0.05)),
                BetSuggestion("trifecta_box", f"{a}={c}={e}", max(rates[0] * rates[2], 0.05)),
                BetSuggestion("trifecta", f"{a}->{d}->{b}", max(rates[0] * 0.3, 0.05)),
                BetSuggestion("wide", f"{a}-{d}", max(rates[0] * 0.5, 0.05)),
            ]
        )
    return templates


def _allocate_budget(budget: float, suggestions: List[BetSuggestion], policy: str) -> List[float]:
    if not suggestions:
        return []
    if policy == "inv-odds":
        weights = [1.0 / max(s.probability, 1e-4) for s in suggestions]
    elif policy == "kelly":
        weights = [s.probability * (1 - s.probability) for s in suggestions]
    else:
        weights = [1.0 for _ in suggestions]
    total = sum(weights) or 1.0
    return [budget * (w / total) for w in weights]


def suggest_bets(
    predictions: List[Dict[str, Any]],
    zone_thresholds: Dict[str, Any],
    budget: int,
    bet_policy: str = "flat",
    max_points: Optional[int] = None,
) -> List[Dict[str, Any]]:
    grouped = _group_by_race(predictions)
    race_ids = list(grouped.keys())
    per_race_budget = budget / max(len(race_ids), 1)
    outputs: List[Dict[str, Any]] = []

    for race_id, entries in grouped.items():
        entries = sorted(entries, key=lambda r: r.get("top3_prob", 0), reverse=True)
        abc_rows = entries[:3]
        extras = [row["rider_name"] for row in entries[3:]]
        a_rate = float(abc_rows[0].get("top3_prob", 0.0))
        b_rate = float(abc_rows[1].get("top3_prob", 0.0)) if len(abc_rows) > 1 else 0.0
        c_rate = float(abc_rows[2].get("top3_prob", 0.0)) if len(abc_rows) > 2 else 0.0
        ct_value = float(abc_rows[0].get("CT_value", a_rate * b_rate * c_rate))
        zone = classify_zone(a_rate, ct_value, zone_thresholds)
        suggestions = _zone_templates(
            zone,
            [row["rider_name"] for row in abc_rows],
            [a_rate, b_rate, c_rate],
            extras,
            ct_value,
        )
        if max_points is not None:
            suggestions = suggestions[:max_points]
        allocations = _allocate_budget(per_race_budget, suggestions, bet_policy)
        allocation_rows = [
            {"bet_type": s.bet_type, "pattern": s.pattern, "stake": round(stake, 2)}
            for s, stake in zip(suggestions, allocations)
        ]
        outputs.append(
            {
                "race_id": race_id,
                "zone": zone,
                "A": abc_rows[0]["rider_name"],
                "B": abc_rows[1]["rider_name"] if len(abc_rows) > 1 else "",
                "C": abc_rows[2]["rider_name"] if len(abc_rows) > 2 else "",
                "A_rate": a_rate,
                "B_rate": b_rate,
                "C_rate": c_rate,
                "CT_value": ct_value,
                "bet_type": "multi",
                "pattern": zone,
                "n_tickets": len(allocation_rows),
                "budget": per_race_budget,
                "allocation_json": dump_json(allocation_rows),
            }
        )
    return outputs
# --- simulate_bets: CLI用ダミー実装 ---
import pandas as pd
from pathlib import Path

def simulate_bets(df: pd.DataFrame, budget: int = 10000, policy: str = "flat") -> pd.DataFrame:
    """ダミーの買い目シミュレーション。予測スコア上位3人を選択。"""
    if "pred_score" not in df.columns:
        df["pred_score"] = 0.5
    out = []
    for race_id, group in df.groupby("race_id" if "race_id" in df.columns else df.index):
        sel = group.nlargest(3, "pred_score")
        sel = sel.assign(bet=budget / 3, expected_return=sel["pred_score"] * (budget / 3))
        out.append(sel)
    return pd.concat(out)

