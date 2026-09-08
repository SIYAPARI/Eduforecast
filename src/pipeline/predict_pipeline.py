from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import dill
import joblib
import pandas as pd
import numpy as np


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gender": self.gender,
                    "race_ethnicity": self.race_ethnicity,
                    "parental_level_of_education": self.parental_level_of_education,
                    "lunch": self.lunch,
                    "test_preparation_course": self.test_preparation_course,
                    "reading_score": float(self.reading_score),
                    "writing_score": float(self.writing_score),
                }
            ]
        )


class Predict_Pipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_path = self._resolve_model_path()
        self.preprocessor_path = self._resolve_preprocessor_path()
        self._load_artifacts()

    def _resolve_model_path(self) -> str:
        project_root = Path(__file__).resolve().parents[2]
        possible_paths = [
            project_root / "artifacts" / "model.pkl",
            project_root / "artifacts" / "student_model.pkl",
            project_root / "src" / "model" / "model.pkl",
            project_root / "model.pkl",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "No trained model found. Please export a model into artifacts/model.pkl."
        )

    def _resolve_preprocessor_path(self) -> str:
        project_root = Path(__file__).resolve().parents[2]
        possible_paths = [
            project_root / "artifacts" / "preprocessor.pkl",
            project_root / "artifacts" / "preprocessor_obj.pkl",
            project_root / "src" / "model" / "preprocessor.pkl",
            project_root / "preprocessor.pkl",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "No preprocessor found. Please export a preprocessor into artifacts/preprocessor.pkl."
        )

    def _load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
        except Exception:
            with open(self.model_path, "rb") as f:
                self.model = dill.load(f)

        try:
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = dill.load(f)
        except Exception:
            self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model or preprocessor not loaded.")
        scaled_features = self.preprocessor.transform(df)
        prediction = self.model.predict(scaled_features)
        return np.asarray(prediction).ravel()

    def _education_score(self, value: str) -> float:
        mapping = {
            "some high school": 1,
            "high school": 2,
            "some college": 3,
            "associate's degree": 4,
            "bachelor's degree": 5,
            "master's degree": 6,
        }
        return mapping.get(str(value).strip().lower(), 2)

    def _race_score(self, value: str) -> float:
        mapping = {
            "group a": 1,
            "group b": 2,
            "group c": 3,
            "group d": 4,
            "group e": 5,
        }
        return mapping.get(str(value).strip().lower(), 3)

    def _compute_feature_contributions(self, row: pd.Series) -> Dict[str, Dict[str, float]]:
        reading_score = float(row.get("reading_score", 0))
        writing_score = float(row.get("writing_score", 0))
        test_prep = str(row.get("test_preparation_course", "")).strip().lower()
        lunch = str(row.get("lunch", "")).strip().lower()
        parental_edu = str(row.get("parental_level_of_education", "")).strip().lower()
        race = str(row.get("race_ethnicity", "")).strip().lower()
        gender = str(row.get("gender", "")).strip().lower()

        raw_scores = {
            "Writing Score": (writing_score / 100.0) * 18.0 - 3.0,
            "Reading Score": (reading_score / 100.0) * 14.0 - 2.0,
            "Test Prep": 6.0 if test_prep in ["completed", "yes", "finished"] else -2.5,
            "Parental Education": self._education_score(parental_edu) * 2.0 - 2.0,
            "Lunch": 4.0 if lunch in ["standard", "free/reduced"] else -1.5,
            "Race / Ethnicity": self._race_score(race) * 1.5 - 2.0,
            "Gender": 2.0 if gender in ["female", "f"] else -0.5,
        }

        total_abs = sum(abs(v) for v in raw_scores.values()) or 1.0

        feature_contributions = {}
        for name, impact in raw_scores.items():
            percentage = abs(impact) / total_abs * 100.0
            feature_contributions[name] = {
                "impact": round(float(impact), 2),
                "percentage": round(float(max(8.0, min(100.0, percentage))), 2),
            }

        return feature_contributions

    def _compute_confidence(self, row: pd.Series, prediction: float) -> int:
        reading_score = float(row.get("reading_score", 0))
        writing_score = float(row.get("writing_score", 0))
        test_prep = str(row.get("test_preparation_course", "")).strip().lower()

        score_density = (reading_score + writing_score) / 200.0
        confidence = 55 + (score_density * 25.0)

        if test_prep in ["completed", "yes", "finished"]:
            confidence += 8

        if 60 <= prediction <= 90:
            confidence += 8

        if prediction < 35 or prediction > 95:
            confidence -= 10

        return int(max(40, min(96, round(confidence))))

    def explain_prediction(self, df: pd.DataFrame) -> dict:
        prediction = float(self.predict(df)[0])
        row = df.iloc[0]

        feature_contributions = self._compute_feature_contributions(row)
        confidence_pct = self._compute_confidence(row, prediction)

        ranked_drivers = sorted(
            feature_contributions.items(),
            key=lambda item: abs(item[1]["impact"]),
            reverse=True
        )[:3]

        top_drivers = [name for name, _ in ranked_drivers]

        return {
            "prediction": round(prediction, 2),
            "confidence_pct": confidence_pct,
            "top_drivers": top_drivers,
            "feature_contributions": feature_contributions,
        }