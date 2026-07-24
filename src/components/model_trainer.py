import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {

                "Linear Regression": LinearRegression(),

                "K-Neighbors Regressor": KNeighborsRegressor(),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),

                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "XGBoost Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0
                ),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                ),

                "AdaBoost": AdaBoostRegressor(
                    random_state=42
                )

            }

            params = {

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11]
                },

                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ],
                    "splitter": [
                        "best",
                        "random"
                    ],
                    "max_depth": [
                        None,
                        5,
                        10,
                        20,
                        30
                    ]
                },

                "Random Forest": {
                    "n_estimators": [
                        50,
                        100,
                        200
                    ],
                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ],
                    "max_depth": [
                        None,
                        5,
                        10,
                        20
                    ]
                },

                "Gradient Boosting": {
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],
                    "subsample": [
                        0.6,
                        0.8,
                        1.0
                    ],
                    "n_estimators": [
                        50,
                        100,
                        200
                    ],
                    "max_depth": [
                        3,
                        5
                    ]
                },

                "XGBoost Regressor": {
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],
                    "n_estimators": [
                        50,
                        100,
                        200
                    ],
                    "max_depth": [
                        3,
                        5,
                        7
                    ]
                },

                "CatBoost Regressor": {
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],
                    "iterations": [
                        50,
                        100,
                        200
                    ],
                    "depth": [
                        4,
                        6,
                        8
                    ]
                },

                "AdaBoost": {
                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],
                    "n_estimators": [
                        50,
                        100,
                        200
                    ]
                }

            }

            logging.info("Evaluating all models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            print(model_report)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException(
                    "No best model found.",
                    sys
                )

            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"Best R2 Score : {best_model_score}")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score : {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)