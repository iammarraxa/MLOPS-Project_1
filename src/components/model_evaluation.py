import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score
from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.entity.s3_estimator import Proj1Estimator

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_best_model(self) -> Optional[Proj1Estimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name, model_path)

            if proj1_estimator.is_model_present(model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys) from e
        
    def __map_gender_column(self, df: pd.DataFrame):
        logging.info('Mapping "Gender" column to binary values')
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df
    
    def _create_dummy_columns(self, df: pd.DataFrame):
        logging.info('Creating dummy variables for categoricla features')
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_columns(self, df: pd.DataFrame):
        logging.info('Renaming specific columns & casting to int')
        df = df.rename(columns={
            'Vehicle_Age_< 1 Year': 'Vehicle_Age_lt_1_Year',
            'Vehicle_Age_> 2 Years': 'Vehicle_Age_gt_2_Years'
        })
        for col in ['Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes']:
            if col in df.columns:
                df = df.drop('_id', axis=1)
            return df
        
    def _drop_id_column(self, df: pd.DataFrame):
        logging.info('Dropping "id" column')
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        return df
    
    def evaluate_model(self) -> EvaluateModelResponse:
        '''evaluates trained model with production model & chooses the best model'''
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            logging.info('Test data loaded & mow transforming it for predictions')

            x = self.__map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)
            
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info('Trained model loaded/exists')
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f'f1 score for this model: {trained_model_f1_score}')

            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is None:
                logging.info(f'Computing f1 score for production model')
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f'f1 score prodcution model: {best_model_f1_score}    |    f1 score new trained model: {trained_model_f1_score}')

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score, best_model_f1_score,
                                           trained_model_f1_score > tmp_best_model_score,
                                           trained_model_f1_score - tmp_best_model_score)
            logging.info(f'Result: {result}')
            return result
        except Exception as e:
            raise MyException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            print('________________________________________________________________________________________________________________')
            logging.info('Initialized Model Evaluation Artifact')
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(evaluate_model_response.is_model_accepted,
                                                                evaluate_model_response.difference,
                                                                s3_model_path,
                                                                self.model_trainer_artifact.trained_model_file_path)
            logging.info(f'Model evaluation artifact: {model_evaluation_artifact}')
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e