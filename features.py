"""Feature extraction utilities without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

NUMERIC_FEATURES = [
    "score",
    "score_roll_mean",
    "backs",
    "backs_roll_mean",
    "homes",
    "homes_roll_mean",
    "starts",
    "starts_roll_mean",
    "win_rate",
    "win_rate_roll_mean",
    "quinella_rate",
    "quinella_rate_roll_mean",
    "top3_rate",
    "top3_rate_roll_mean",
    "kimarite_nige",
    "kimarite_nige_roll_mean",
    "kimarite_makuri",
    "kimarite_makuri_roll_mean",
    "kimarite_sashi",
    "kimarite_sashi_roll_mean",
    "kimarite_mark",
    "kimarite_mark_roll_mean",
    "age",
    "style_escape",
    "style_chaser",
    "style_both",
    "style_flex",
    "style_unknown",
]

CATEGORICAL_FEATURES = ["track", "class", "grade", "prefecture"]


@dataclass
class FeatureBuilder:
    categories: Dict[str, List[str]] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def fit(self, rows: Sequence[Dict[str, object]]) -> None:
        for feature in CATEGORICAL_FEATURES:
            seen: List[str] = []
            for row in rows:
                value = row.get(feature)
                if value is None or value == "":
                    continue
                value_str = str(value)
                if value_str not in seen:
                    seen.append(value_str)
            self.categories[feature] = seen
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = list(NUMERIC_FEATURES)
        for feature in CATEGORICAL_FEATURES:
            for category in self.categories.get(feature, []):
                names.append(f"{feature}={category}")
        return names

    def transform(self, rows: Sequence[Dict[str, object]]) -> List[List[float]]:
        transformed: List[List[float]] = []
        for row in rows:
            features: List[float] = []
            for name in NUMERIC_FEATURES:
                value = row.get(name)
                features.append(float(value) if isinstance(value, (int, float)) else 0.0)
            for feature in CATEGORICAL_FEATURES:
                cats = self.categories.get(feature, [])
                value = str(row.get(feature) or "")
                for category in cats:
                    features.append(1.0 if value == category else 0.0)
            transformed.append(features)
        return transformed

    def fit_transform(self, rows: Sequence[Dict[str, object]]) -> List[List[float]]:
        self.fit(rows)
        return self.transform(rows)

