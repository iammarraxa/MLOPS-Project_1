import sys
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_np_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ClassificationMetricArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_tansformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        '''
        Trains RandomForestClassifier with specified parameters

        Output  |   Metric artifact object & Trained model object
        '''
        try:
            logging.info('Training RFC with with specified parameters')
            X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]
            logging.info('train/test split done')

            model = RandomForestClassifier(
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state
            )

            logging.info('Model training going on......')
            model.fit(X_train, y_train)
            logging.info('Model training done')

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            metric_artifact = ClassificationMetricArtifact(accuracy, f1, precision, recall)
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info('Entered initiate_model_trainer method of ModelTrainer class')
        '''
        Output  |   Model Trainer Artifact
        '''
        try:
            print('_____________________________________________________________________________')
            print('Starting Model Trainer Component')

            train_arr = load_np_array_data(self.data_tansformation_artifact.transformed_train_file_path)
            test_arr = load_np_array_data(self.data_tansformation_artifact.transformed_test_file_path)
            logging.info('train/test data loaded')

            trained_model, metric_artifact = self.get_model_object_and_report(train_arr, test_arr)
            logging.info('Model object & artifact loaded')

            preprocessing_obj = load_object(self.data_tansformation_artifact.transformed_object_file_path)
            logging.info('Preprocessing object loaded')

            if accuracy_score(train_arr[:,-1], trained_model.predict(train_arr[:,:-1])) < self.model_trainer_config.expected_accuracy:
                logging.info('No model found with score above the base score')
                raise Exception('No model found with score above the base score')
            
            logging.info('Saving new model as performance is better than the previous one')
            my_model = MyModel(preprocessing_obj, trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info('Saved final model object includes both preprocessing & trained model')

            model_trainer_artifact = ModelTrainerArtifact(
                self.model_trainer_config.trained_model_file_path,
                metric_artifact
            )
            logging.info(f'Model trainer artifact: {model_trainer_artifact}')
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

