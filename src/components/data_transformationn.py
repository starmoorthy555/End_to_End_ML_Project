import os
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transfomation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining processing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'math_score'
            numerical_columns = ['writing_score','reading_score']

            input_features_train_data = train_data.drop([target_column],axis=1)
            target_features_train_data = train_data[target_column]

            input_features_test_data = test_data.drop([target_column],axis=1)
            target_features_test_data = test_data[target_column]

            logging.info(
                f'Appliying processing_object on training DatFrame and testing DataFrame'
                         )
            
            input_features_train_data_arr = preprocessing_obj.fit_transform(input_features_train_data)
            input_features_test_data_arr = preprocessing_obj.fit_transform(input_features_test_data)

            train_arr = np.c_[
                input_features_train_data_arr,np.array(target_features_train_data)
            ]

            test_arr = np.c_[
                input_features_test_data_arr,np.array(target_features_test_data)
            ]

            logging.info(f'Saving Processing object')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e,sys)
    





























