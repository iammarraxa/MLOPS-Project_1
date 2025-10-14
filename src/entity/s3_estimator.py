import sys
from pandas import DataFrame
from src.exception import MyException
from src.entity.estimator import MyModel
from src.cloud_storage.aws_storage import SimpleStorageService

class Proj1Estimator:
    '''saves & retrieves model from s3 bucket and then do prediction'''
    def __init__(self, bucket_name, model_path):
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(self.bucket_name, model_path)
        except MyException as e:
            print(e)
            return False
        
    def load_model(self) -> MyModel:
        return self.s3.load_model(self.model_path, self.bucket_name)
    
    def save_model(self, from_file, remove: bool = False) -> None:
        try:
            self.s3.upload_file(from_file, self.model_path, self.bucket_name, remove)
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys) from e