# -*- coding: utf-8 -*-
"""Model training and prediction using plain Python + a thin class wrapper."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # ラッパークラス用に追加

import bets
import data
from features import FeatureBuilder
from simulate import compute_ct
from utils import ensure_directory, get_logger

logger = get_logger(__name__)


def sigmoid(x: float) -> float:
    if x < -50:
        return 1e-6
    if x > 50:
        return 1 - 1e-6
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class LogisticModel:
    weights: List[float]
    bias: float

    def predict_proba(self, features: List[List[float]]) -> List[float]:
        probs: List[float] = []
        for row in features:
            score = sum(w * x for w, x in zip(self.weights, row)) + self.bias
            probs.append(sigmoid(score))
        return probs

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LogisticModel":
        return cls(weights=list(payload["weights"]), bias=float(payload["bias"]))


@dataclass
class ModelArtifacts:
    top3_model: LogisticModel
    win_model: LogisticModel
    categories: Dict[str, List[str]]
    feature_names: List[str]
    metadata: Dict[str, Any]

    def save(self, directory: str | Path) -> None:
        ensure_directory(directory)
        payload = {
            "top3_model": self.top3_model.to_dict(),
            "win_model": self.win_model.to_dict(),
            "categories": self.categories,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        with open(Path(directory) / "model.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "ModelArtifacts":
        with open(Path(directory) / "model.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            top3_model=LogisticModel.from_dict(payload["top3_model"]),
            win_model=LogisticModel.from_dict(payload["win_model"]),
            categories=payload["categories"],
            feature_names=payload["feature_names"],
            metadata=payload.get("metadata", {}),
        )


def _train_logistic(features: List[List[float]], labels: List[Optional[float]], epochs: int = 200) -> LogisticModel:
    if not features:
        return LogisticModel(weights=[], bias=0.0)
    n_features = len(features[0])
    weights = [0.0] * n_features
    bias = 0.0
    learning_rate = 0.05
    l2 = 1e-4

    for epoch in range(epochs):
        for x, y in zip(features, labels):
            if y is None:
                continue
            score = sum(w * xi for w, xi in zip(weights, x)) + bias
            pred = sigmoid(score)
            error = pred - y
            for i in range(n_features):
                weights[i] -= learning_rate * (error * x[i] + l2 * weights[i])
            bias -= learning_rate * error
    return LogisticModel(weights=weights, bias=bias)


def config_thresholds(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = json.loads(json.dumps(bets.DEFAULT_THRESHOLDS))
    if overrides:
        for zone, values in overrides.items():
            if zone in merged and isinstance(values, dict):
                merged[zone].update(values)
            else:
                merged[zone] = values
    return merged


def _write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_directory(Path(path).parent)
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = [str(row.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")


def train(races_path: str | Path, out_dir: str | Path, config: Optional[Dict[str, Any]] = None) -> ModelArtifacts:
    """ファイルパスを受け取って学習（既存の関数API・後方互換用）"""
    config = config or {}
    rows = data.load_races(races_path)
    dataset = data.prepare_training_data(rows, rolling_window=config.get("rolling_window", 5))

    builder = FeatureBuilder()
    feature_matrix = builder.fit_transform(dataset.rows)
    top3_model = _train_logistic(feature_matrix, dataset.labels_top3)
    win_model = _train_logistic(feature_matrix, dataset.labels_win)

    metadata = {
        "races_path": str(races_path),
        "n_rows": len(rows),
        "config": config,
    }

    artifacts = ModelArtifacts(
        top3_model=top3_model,
        win_model=win_model,
        categories=builder.categories,
        feature_names=builder.feature_names,
        metadata=metadata,
    )
    artifacts.save(out_dir)
    logger.info("Model saved to %s", out_dir)
    return artifacts


def _predict_internal(
    artifacts: ModelArtifacts,
    cards: List[Dict[str, Any]],
    ct_method: str,
    mc_iters: int,
    thresholds: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = data.prepare_inference_data(cards)
    builder = FeatureBuilder(categories=dict(artifacts.categories))
    builder.feature_names = list(artifacts.feature_names)
    features = builder.transform(rows)

    top3_probs = artifacts.top3_model.predict_proba(features)
    win_probs = artifacts.win_model.predict_proba(features)

    enriched: List[Dict[str, Any]] = []
    for row, top3, win in zip(rows, top3_probs, win_probs):
        enriched_row = dict(row)
        enriched_row["top3_prob"] = top3
        enriched_row["win_prob"] = win
        enriched.append(enriched_row)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in enriched:
        grouped.setdefault(str(row.get("race_id")), []).append(row)

    results: List[Dict[str, Any]] = []
    for race_id, entries in grouped.items():
        entries.sort(key=lambda r: r.get("top3_prob", 0.0), reverse=True)
        a_rate = float(entries[0].get("top3_prob", 0.0)) if entries else 0.0
        b_rate = float(entries[1].get("top3_prob", 0.0)) if len(entries) > 1 else 0.0
        c_rate = float(entries[2].get("top3_prob", 0.0)) if len(entries) > 2 else 0.0
        ct_value = compute_ct([a_rate, b_rate, c_rate], method=ct_method, mc_iters=mc_iters)
        zone = bets.classify_zone(a_rate, ct_value, thresholds)
        for idx, entry in enumerate(entries, start=1):
            record = dict(entry)
            record["rank_by_top3"] = idx
            record["is_A"] = idx == 1
            record["is_B"] = idx == 2
            record["is_C"] = idx == 3
            record["A_rate"] = a_rate
            record["B_rate"] = b_rate
            record["C_rate"] = c_rate
            record["CT_value"] = ct_value
            record["zone"] = zone
            results.append(record)
    return results


def predict(
    cards: str | Path | List[Dict[str, Any]],
    model_dir: str | Path,
    out_path: Optional[str | Path] = None,
    ct_method: str = "independent",
    mc_iters: int = 2000,
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """ファイルパスや行配列を受け取って予測（既存の関数API・後方互換用）"""
    if isinstance(cards, (str, Path)):
        card_rows = data.load_cards(cards)
    else:
        card_rows = list(cards)
    artifacts = ModelArtifacts.load(model_dir)
    threshold_cfg = config_thresholds(thresholds)
    predictions = _predict_internal(artifacts, card_rows, ct_method, mc_iters, threshold_cfg)
    if out_path:
        _write_csv(out_path, predictions)
        logger.info("Predictions saved to %s", out_path)
    return predictions


def backtest(
    races_path: str | Path,
    model_dir: str | Path,
    budget: int = 10000,
    bet_policy: str = "flat",
    zone_filter: str = "any",
) -> Dict[str, Any]:
    """既存の関数API・後方互換用"""
    races = data.load_races(races_path)
    cards = [dict(row) for row in races]
    for row in cards:
        row.pop("finish_pos", None)
    predictions = predict(cards, model_dir)

    actual_map: Dict[tuple, float] = {}
    for row in races:
        key = (row.get("race_id"), row.get("rider_name"))
        actual_map[key] = float(row.get("finish_pos") or 99)

    for row in predictions:
        key = (row.get("race_id"), row.get("rider_name"))
        row["finish_pos"] = actual_map.get(key)
        row["hit"] = 1.0 if (row["finish_pos"] and row["finish_pos"] <= 3) else 0.0

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get("race_id")), []).append(row)

    accuracy_values: List[float] = []
    for race_id, entries in grouped.items():
        entries.sort(key=lambda r: r.get("rank_by_top3", 99))
        top3 = entries[:3]
        if not top3:
            continue
        accuracy_values.append(sum(r.get("hit", 0.0) for r in top3) / len(top3))

    threshold_cfg = config_thresholds()
    bets_plan = bets.suggest_bets(predictions, threshold_cfg, budget=budget, bet_policy=bet_policy)
    if zone_filter != "any":
        bets_plan = [b for b in bets_plan if b["zone"] == zone_filter]

    summary = {
        "races": len(grouped),
        "top3_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0,
        "avg_budget": sum(b["budget"] for b in bets_plan) / len(bets_plan) if bets_plan else 0.0,
    }
    logger.info("Backtest summary: %s", summary)
    return summary


# -----------------------------------------------------------------------------
# ここから追加: 薄いラッパークラス（main.py の Model 互換API）
# -----------------------------------------------------------------------------
class Model:
    """Thin wrapper to keep main.py's class-based API, backed by ModelArtifacts."""

    def __init__(self, artifacts: Optional[ModelArtifacts] = None):
        self.artifacts: Optional[ModelArtifacts] = artifacts

    def train(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """DataFrame から学習し、self.artifacts を構築"""
        config = config or {}
        # DataFrame -> rows(dict)
        rows = [dict(r) for r in df.to_dict(orient="records")]
        dataset = data.prepare_training_data(rows, rolling_window=config.get("rolling_window", 5))

        builder = FeatureBuilder()
        feature_matrix = builder.fit_transform(dataset.rows)
        top3_model = _train_logistic(feature_matrix, dataset.labels_top3)
        win_model = _train_logistic(feature_matrix, dataset.labels_win)

        metadata = {"n_rows": len(rows), "config": config}
        self.artifacts = ModelArtifacts(
            top3_model=top3_model,
            win_model=win_model,
            categories=builder.categories,
            feature_names=builder.feature_names,
            metadata=metadata,
        )
        logger.info("Training completed: rows=%d, features=%d",
                    len(rows), len(builder.feature_names))

    def save(self, out_dir: str | Path) -> None:
        if not self.artifacts:
            raise RuntimeError("Model artifacts are empty. Train first.")
        self.artifacts.save(out_dir)
        logger.info("Model saved to %s", out_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "Model":
        artifacts = ModelArtifacts.load(model_dir)
        logger.info("Model loaded from %s", model_dir)
        return cls(artifacts)

    def predict(
        self,
        df: pd.DataFrame,
        ct_method: str = "independent",
        mc_iters: int = 2000,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """DataFrame を受け取り予測行 DataFrame を返す"""
        if not self.artifacts:
            raise RuntimeError("Model not loaded/trained.")
        rows = [dict(r) for r in df.to_dict(orient="records")]
        preds = _predict_internal(
            self.artifacts,
            rows,
            ct_method=ct_method,
            mc_iters=mc_iters,
            thresholds=config_thresholds(thresholds),
        )
        return pd.DataFrame(preds)
# --- 互換用ラッパークラス追加 ---
import pandas as pd

class Model:
    def __init__(self):
        pass

    def train(self, df: pd.DataFrame):
        print("Model.train() called – dummy placeholder")

    def save(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "model.pkl").write_text("dummy model")

    @classmethod
    def load(cls, model_dir: str):
        print("Model.load() called – dummy placeholder")
        return cls()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Model.predict() called – dummy placeholder")
        df["pred_score"] = 0.5
        return df

