import os
from dataclasses import dataclass
import sys

from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_filpath = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfiq = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        logging.info('Initiate train test split')

        x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
        models ={
            "Random_forest":RandomForestRegressor(),
            "Adabost":AdaBoostRegressor(),
            "Gradient_boost":GradientBoostingRegressor(),
            "Linear_regresser":LinearRegression(),
            "Kneighbour":KNeighborsRegressor(),
            "tree":DecisionTreeRegressor(),
        }

        model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                             models=models)
        

        best_model_score = max(sorted(model_report.values()))
        print(best_model_score)

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]
        print(best_model)

        if best_model_score<0.6:
            raise CustomException("No best model found")

        logging.info(f'Best model found for training and resting data')

        save_object(
            file_path = self.modeltrainerconfiq.trained_model_filpath,
            obj = best_model
        )

        predicted = best_model.predict(x_test)
        return r2_score(predicted,y_test)



































