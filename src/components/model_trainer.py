import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train , y_train , X_test, y_test = (
                train_arr[:,:-1],# all rows , all columns except last column
                train_arr[:,-1], # all rows , last column
                test_arr[:,:-1], # all rows , all columns except last column
                test_arr[:,-1]   # all rows , last column
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor() 
            }
            model_report: dict  = evaluate_models (X_train=X_train, y_train=y_train,
                                                   X_test=X_test, y_test=y_test, models=models)
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            #to get model name from model report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model= models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted= best_model.predict(X_test)
            final_r2_score= r2_score(y_test, predicted)
            return final_r2_score


        except Exception as e:
            raise CustomException(e, sys)
    