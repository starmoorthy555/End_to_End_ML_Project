import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
import os

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','data.csv')

class DataInjection:
    def __init__(self):
        self.injection_config = DataIngestionConfig()

    def initiate_data_injection(self):
        logging.info("Data injction initiate")
        try:
            df = pd.read_csv("notebook\data\data.csv")
            logging.info("Read the Dataset as DataFrame")

            os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)

            df.to_csv(self.injection_config.raw_data_path,index=False,header=True)

            train_set,test_set = train_test_split(df,test_size=0.4,random_state=42)
            logging.info("train_test_split initiated")

            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.injection_config.test_data_path,index=False,header=True)

            logging.info("Injection of the data is complited")

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path   
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataInjection() 
    obj.initiate_data_injection()







    


