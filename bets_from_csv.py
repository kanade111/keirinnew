#!/usr/bin/env python
"""Utility for creating betting tickets from race cards and model predictions."""
from __future__ import annotations

import argparse
import itertools
import logging
import math
from pathlib import Path
import sys
from typing import Iterable, List, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    print(f"エラー: 必須ライブラリの読み込みに失敗しました: {exc}", file=sys.stderr)
    sys.exit(1)

LOGGER = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "race_id",
    "race_name",
    "venue",
    "bet_type",
    "combination",
    "stake",
    "ev",
    "rank",
    "notes",
]

CARD_EXPECTED_COLUMNS = {
    "race_id",
    "date",
    "race_no",
    "stadium",
    "track",
    "class",
    "lane_no",
    "rider_name",
    "score",
    "style",
    "line_id",
    "line_pos",
}

PRED_EXPECTED_COLUMNS = {
    "race_id",
    "lane_no",
    "p_win",
    "p_place",
    "p_show",
    "odds_win",
    "odds_quinella",
    "odds_trifecta",
    "score",
}

ENCODINGS_TO_TRY = ["utf-8", "cp932", "euc_jp", "utf-8-sig"]


def read_csv_safely(path: Path, expected_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read CSV with multiple encodings to avoid mojibake."""

    path = Path(path)
    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            LOGGER.debug("Trying to read %s with encoding %s", path, encoding)
            df = pd.read_csv(path, encoding=encoding)
            df = _maybe_fix_header(df, path, encoding, expected_columns)
            LOGGER.info("Loaded %s (%d rows) with encoding %s", path.name, len(df), encoding)
            return df
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for fallback
            last_error = exc
            LOGGER.debug("Failed reading %s with %s: %s", path, encoding, exc)
            continue
    if last_error is None:
        raise RuntimeError(f"CSV読み込み失敗: {path}")
    raise RuntimeError(f"CSV読み込み失敗: {path}: {last_error}")


def _maybe_fix_header(
    df: pd.DataFrame,
    path: Path,
    encoding: str,
    expected_columns: Iterable[str] | None,
) -> pd.DataFrame:
    """Attempt to recover when the first row actually contains headers."""

    if expected_columns is None or df.empty:
        return df

    normalized_columns = [str(col).strip() for col in df.columns]
    df.columns = normalized_columns
    expected = {col for col in expected_columns if col}
    if expected.intersection(df.columns):
        return df

    first_row = df.iloc[0].astype(str).str.strip()
    overlap = expected.intersection(set(first_row))
    if not overlap:
        return df

    LOGGER.debug("Detected header row in data when reading %s with %s", path, encoding)
    df_no_header = pd.read_csv(path, encoding=encoding, header=None)
    header = [str(value).strip() for value in df_no_header.iloc[0].tolist()]
    df_no_header = df_no_header.iloc[1:].reset_index(drop=True)
    df_no_header.columns = header
    return df_no_header


def _normalize_lane(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            as_float = float(text)
            value = as_float
        except ValueError:
            return text
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isnan(num):
        return None
    if abs(num - round(num)) < 1e-6:
        return str(int(round(num)))
    return str(num).rstrip("0").rstrip(".")


def _prepare_probability_series(df: pd.DataFrame, column: str, fallback: Sequence[float] | None = None) -> pd.Series:
    if column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        if np.isfinite(series).all() and series.sum() > 0:
            return series
        if series.max() > 0:
            return series
    if fallback is not None:
        return pd.Series(fallback, index=df.index, dtype=float)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def _ranking_key(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "p_win",
        "p_place",
        "score_pred",
        "score_card",
        "p_show",
    ]
    for column in candidates:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                filled = series.fillna(0.0)
                max_value = filled.max()
                min_value = filled.min()
                if max_value > min_value:
                    return (filled - min_value) / (max_value - min_value + 1e-12)
                if max_value > 0:
                    return filled
    if len(df) == 0:
        return pd.Series(dtype=float)
    descending = np.linspace(1.0, 0.0, len(df), endpoint=False)
    return pd.Series(descending, index=df.index)


def _calculate_two_lane_combination(
    axis_row: pd.Series,
    opponent_row: pd.Series,
    axis_rank: int,
    top_prob: float,
    second_prob: float,
) -> dict | None:
    p_axis = float(axis_row.get("prob_win", 0.0) or 0.0)
    if p_axis <= 0:
        return None

    opp_place = opponent_row.get("prob_place")
    if opp_place is None or not np.isfinite(opp_place):
        opp_place = opponent_row.get("prob_win", 0.0)
    opp_place = float(opp_place or 0.0)
    denom = max(1e-6, 1.0 - p_axis)
    conditional = min(1.0, max(0.0, opp_place) / denom)
    probability = p_axis * conditional
    if probability <= 0:
        return None

    odds_candidates: List[float] = []
    for column in ("odds_quinella", "odds_win"):
        value = axis_row.get(column)
        if value is None:
            value = opponent_row.get(column)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if numeric and np.isfinite(numeric) and numeric > 0:
            odds_candidates.append(numeric)
    odds = float(np.mean(odds_candidates)) if odds_candidates else 0.0
    if odds <= 0:
        odds = 1.0 / max(probability, 1e-6)

    ev = probability * odds

    note = _derive_note(axis_rank, p_axis, top_prob, second_prob)
    combination = f"{axis_row['lane_no']}-{opponent_row['lane_no']}"
    return {
        "bet_type": "2連単",
        "combination": combination,
        "probability": probability,
        "odds": odds,
        "ev": ev,
        "notes": note,
    }


def _calculate_three_lane_combination(
    ordered_rows: Sequence[pd.Series],
    top_prob: float,
    second_prob: float,
) -> dict | None:
    lane_labels: List[str] = []
    for lane in [row["lane_no"] for row in ordered_rows]:
        text = str(lane).strip()
        try:
            lane_value = float(text)
            if abs(lane_value - round(lane_value)) < 1e-6:
                text = str(int(round(lane_value)))
        except (TypeError, ValueError):
            pass
        lane_labels.append(text)
    lane_str = "-".join(lane_labels)

    p_values = [float(row.get("prob_win", 0.0) or 0.0) for row in ordered_rows]
    if p_values[0] <= 0:
        return None
    denom_remaining = 1.0
    probability = 1.0
    for idx, p_val in enumerate(p_values):
        p_val = max(0.0, min(1.0, p_val))
        if denom_remaining <= 1e-6:
            return None
        conditional = min(1.0, p_val / denom_remaining)
        probability *= conditional
        denom_remaining = max(1e-6, denom_remaining - p_val)
    if probability <= 0:
        return None

    odds_candidates: List[float] = []
    for row in ordered_rows:
        value = row.get("odds_trifecta")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if numeric and np.isfinite(numeric) and numeric > 0:
            odds_candidates.append(numeric)
    odds = float(np.mean(odds_candidates)) if odds_candidates else 0.0
    if odds <= 0:
        odds = 1.0 / max(probability, 1e-6)

    ev = probability * odds
    note = _derive_note(1, p_values[0], top_prob, second_prob)
    return {
        "bet_type": "3連単",
        "combination": lane_str,
        "probability": probability,
        "odds": odds,
        "ev": ev,
        "notes": note,
    }


def _derive_note(axis_rank: int, axis_prob: float, top_prob: float, second_prob: float) -> str:
    if axis_rank <= 1 and axis_prob >= 0.45:
        if axis_prob - second_prob >= 0.1:
            return "本命ライン"
        return "本命"
    if axis_rank <= 2 and axis_prob >= 0.3:
        return "対抗"
    if axis_prob <= max(0.15, top_prob * 0.6):
        return "穴"
    return "ライン重視"


def _allocate_from_weights(weights: Sequence[float], total: int) -> List[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]
    weights_array = np.array(weights, dtype=float)
    weights_array = np.where(np.isfinite(weights_array) & (weights_array > 0), weights_array, 0.0)
    if weights_array.sum() <= 0:
        weights_array = np.ones(len(weights_array), dtype=float)
    scaled = weights_array / weights_array.sum() * total
    floors = np.floor(scaled).astype(int)
    remainder = int(total - floors.sum())
    if remainder > 0:
        residuals = scaled - floors
        order = np.argsort(residuals)[::-1]
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors.tolist()


def _kelly_fractions(probabilities: Sequence[float], odds_list: Sequence[float]) -> List[float]:
    fractions: List[float] = []
    for prob, odds in zip(probabilities, odds_list):
        prob = max(0.0, min(0.999, float(prob)))
        try:
            odds_val = float(odds)
        except (TypeError, ValueError):
            odds_val = 0.0
        if odds_val <= 1.0:
            fractions.append(0.0)
            continue
        b = odds_val - 1.0
        edge = prob - (1.0 - prob) / b
        fraction = max(0.0, edge)
        fraction = float(min(0.05, fraction))
        fractions.append(fraction)
    return fractions


def _build_race_name(row: pd.Series) -> tuple[str, str]:
    stadium = str(row.get("stadium", "")).strip()
    class_name = str(row.get("class", "")).strip()
    race_no_raw = row.get("race_no")
    race_no = ""
    if race_no_raw is not None and not pd.isna(race_no_raw):
        try:
            race_no = f"{int(float(race_no_raw))}R"
        except (TypeError, ValueError):
            race_no = f"{race_no_raw}R"
    components = [stadium]
    if class_name:
        components.append(class_name)
    if race_no:
        components.append(race_no)
    race_name = " ".join(filter(None, components))
    venue = stadium or ""
    return race_name, venue


def plan_bets(
    df_cards: pd.DataFrame,
    df_pred: pd.DataFrame,
    budget: int = 10000,
    policy: str = "flat",
) -> pd.DataFrame:
    cards = df_cards.copy() if df_cards is not None else pd.DataFrame()
    preds = df_pred.copy() if df_pred is not None else pd.DataFrame()

    if "score" in cards.columns:
        cards = cards.rename(columns={"score": "score_card"})
    if "score" in preds.columns:
        preds = preds.rename(columns={"score": "score_pred"})

    for frame in (cards, preds):
        if "race_id" in frame.columns:
            frame["race_id"] = frame["race_id"].astype(str).str.strip()
        if "lane_no" in frame.columns:
            frame["lane_no"] = frame["lane_no"].map(_normalize_lane)

    if {"race_id", "lane_no"} - set(cards.columns):
        LOGGER.warning("cards.csv に race_id もしくは lane_no が不足しています")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if {"race_id", "lane_no"} - set(preds.columns):
        LOGGER.warning("predictions.csv に race_id もしくは lane_no が不足しています")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cards = cards.dropna(subset=["race_id", "lane_no"])
    preds = preds.dropna(subset=["race_id", "lane_no"])

    merged = pd.merge(cards, preds, on=["race_id", "lane_no"], how="inner", suffixes=("_card", "_pred"))
    if merged.empty:
        LOGGER.warning("結合後のデータが空です")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    numeric_columns = [
        "p_win",
        "p_place",
        "p_show",
        "odds_win",
        "odds_quinella",
        "odds_trifecta",
        "prob_win",
        "prob_place",
        "score_card",
        "score_pred",
    ]
    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    sort_columns = [col for col in ("date", "race_no", "race_id") if col in merged.columns]
    if sort_columns:
        merged = merged.sort_values(by=sort_columns)

    bet_rows: List[dict] = []
    race_order: List[str] = []

    for race_id, race_df in merged.groupby("race_id", sort=False):
        race_df = race_df.copy()
        race_df["prob_win"] = _prepare_probability_series(race_df, "p_win")
        if race_df["prob_win"].max() <= 0:
            ranking = _ranking_key(race_df)
            if ranking.sum() > 0:
                race_df["prob_win"] = ranking / ranking.sum()
            else:
                race_df["prob_win"] = 1.0 / max(len(race_df), 1)
        race_df["prob_place"] = _prepare_probability_series(
            race_df,
            "p_place",
            fallback=np.clip(race_df["prob_win"].to_numpy() * 1.2, 0.0, 1.0),
        )
        race_df["prob_show"] = _prepare_probability_series(
            race_df,
            "p_show",
            fallback=np.clip(race_df["prob_win"].to_numpy() * 1.5, 0.0, 1.0),
        )

        race_df = race_df.sort_values("prob_win", ascending=False).reset_index(drop=True)
        race_df["lane_no"] = race_df["lane_no"].astype(str)

        top_prob = float(race_df["prob_win"].iloc[0]) if not race_df.empty else 0.0
        second_prob = float(race_df["prob_win"].iloc[1]) if len(race_df) > 1 else 0.0

        first_row = race_df.iloc[0]
        race_name, venue = _build_race_name(first_row)

        combinations: List[dict] = []
        axis_count = min(2, len(race_df))
        for axis_idx in range(axis_count):
            axis_row = race_df.iloc[axis_idx]
            opponents = race_df.drop(axis_idx).reset_index(drop=True)
            opp_limit = min(3, len(opponents))
            if opp_limit <= 0:
                continue
            opponents = opponents.head(max(2, opp_limit))
            for _, opp_row in opponents.iterrows():
                combo = _calculate_two_lane_combination(axis_row, opp_row, axis_idx + 1, top_prob, second_prob)
                if combo:
                    combo.update(
                        {
                            "race_id": race_id,
                            "race_name": race_name,
                            "venue": venue,
                        }
                    )
                    combinations.append(combo)

        top_three = race_df.head(3)
        if len(top_three) >= 3:
            permutations = list(itertools.permutations(range(len(top_three)), 3))[:10]
            for order in permutations:
                ordered_rows = [top_three.iloc[idx] for idx in order]
                combo = _calculate_three_lane_combination(ordered_rows, top_prob, second_prob)
                if combo:
                    combo.update(
                        {
                            "race_id": race_id,
                            "race_name": race_name,
                            "venue": venue,
                        }
                    )
                    combinations.append(combo)

        combinations = [combo for combo in combinations if combo.get("ev", 0) > 0]
        if not combinations:
            continue
        combinations.sort(key=lambda x: x.get("ev", 0), reverse=True)
        limit = min(6, len(combinations))
        combinations = combinations[:limit]
        for rank_idx, combo in enumerate(combinations, start=1):
            combo["rank"] = rank_idx
            bet_rows.append(combo)

        race_order.append(race_id)

    if not bet_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    bets_df = pd.DataFrame(bet_rows)
    bets_df["stake"] = 0

    unique_races = list(dict.fromkeys(race_order))
    race_count = len(unique_races)
    if race_count == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    total_budget = max(0, int(budget))
    base_budget = total_budget // race_count
    remainder = total_budget % race_count

    for idx, race_id in enumerate(unique_races):
        race_mask = bets_df["race_id"] == race_id
        race_indices = bets_df.index[race_mask].tolist()
        if not race_indices:
            continue
        race_budget = base_budget + (1 if idx < remainder else 0)
        race_subset = bets_df.loc[race_indices]

        if policy == "proportional":
            weights = race_subset["ev"].clip(lower=0).to_numpy()
            stakes = _allocate_from_weights(weights, race_budget)
        elif policy == "kelly":
            fractions = _kelly_fractions(
                race_subset["probability"].to_numpy(),
                race_subset["odds"].to_numpy(),
            )
            sum_fraction = sum(fractions)
            if sum_fraction <= 0:
                stakes = _allocate_from_weights(np.ones(len(race_subset)), race_budget)
            else:
                raw_amounts = np.array(fractions) * race_budget
                cap = int(min(race_budget, math.floor(raw_amounts.sum())))
                if cap <= 0:
                    stakes = [0 for _ in race_subset.index]
                    max_value = float(raw_amounts.max()) if raw_amounts.size else 0.0
                    if raw_amounts.size and max_value > 0 and race_budget > 0:
                        best_idx = int(np.argmax(raw_amounts))
                        stakes[best_idx] = 1
                else:
                    stakes = _allocate_from_weights(raw_amounts, cap)
        else:
            weights = np.ones(len(race_subset))
            stakes = _allocate_from_weights(weights, race_budget)

        bets_df.loc[race_indices, "stake"] = stakes

    bets_df["stake"] = bets_df["stake"].astype(int)
    bets_df["ev"] = bets_df["ev"].astype(float)

    output_df = bets_df[OUTPUT_COLUMNS].copy()
    output_df = output_df.sort_values(["race_id", "rank", "bet_type"]).reset_index(drop=True)
    return output_df


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate betting tickets from CSV files.")
    parser.add_argument("--cards", required=True, type=Path, help="Path to cards CSV")
    parser.add_argument("--pred", required=True, type=Path, help="Path to predictions CSV")
    parser.add_argument("--out", required=True, type=Path, help="Output path for bets CSV")
    parser.add_argument("--budget", type=int, default=10000, help="Total betting budget")
    parser.add_argument(
        "--policy",
        choices=["flat", "proportional", "kelly"],
        default="flat",
        help="Bet sizing policy",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        cards_df = read_csv_safely(args.cards, CARD_EXPECTED_COLUMNS)
        preds_df = read_csv_safely(args.pred, PRED_EXPECTED_COLUMNS)
        bets_df = plan_bets(cards_df, preds_df, budget=args.budget, policy=args.policy)
    except Exception as exc:  # noqa: BLE001
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if bets_df.empty:
        print("ベット候補が生成できませんでした。入力データをご確認ください。")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bets_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"出力ファイル: {args.out}")
    summary = bets_df.sort_values("ev", ascending=False).head(100)
    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", None):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
