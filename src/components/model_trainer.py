import sys
import os
from dataclasses import dataclass

# ✅ Only Regressors — removed all Classifiers
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ All Regressors now
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
    "Linear Regression": {},
    
    # ✅ Add these three
    "Ridge": {'alpha': [0.1, 1.0, 10.0, 100.0]},
    "Lasso": {'alpha': [0.001, 0.01, 0.1, 1.0]},
    "KNN":   {'n_neighbors': [3, 5, 7, 9]},
    
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    },
    "Random Forest": {
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}

            # ✅ evaluate_models handles training internally — no second loop needed
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ best_model already trained inside evaluate_models
            prediction = best_model.predict(X_test)
            r2_square = r2_score(y_test, prediction)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)