from pathlib import Path

dir = 'src'

list_of_files = {
    f'{dir}/__init.py',
    f'{dir}/components/__init__.py',
    f'{dir}/components/data_ingestion.py',
    f'{dir}/components/data_validation.py',
    f'{dir}/components/data_transformation.py',
    f'{dir}/components/model_trainer.py',
    f'{dir}/components/model_evaluation.py',
    f'{dir}/components/model_pusher.py',
    f'{dir}/configuration/__init__.py',
    f'{dir}/configuration/mongo_db_connection.py',
    f'{dir}/configuration/aws_connection.py',
    f'{dir}/cloud_storage/__init__.py',
    f'{dir}/cloud_storage/aws_storage.py',
    f'{dir}/data_access/__init__.py',
    f'{dir}/data_access/proj1_data.py',
    f'{dir}/constants/__init__.py',
    f'{dir}/entity/__init__.py',
    f'{dir}/entity/config_entity.py',
    f'{dir}/entity/artifact_entity.py',
    f'{dir}/entity/estimator.py',
    f'{dir}/entity/s3_estimator.py',
    f'{dir}/exception/__init__.py',
    f'{dir}/logger/__init__.py',
    f'{dir}/pipeline/__init__.py',
    f'{dir}/pipeline/training_pipeline.py',
    f'{dir}/pipeline/prediction_pipeline.py',
    f'{dir}/utils/__init__.py',
    f'{dir}/utils/main_utils.py',
    'app.py',
    'requirements.txt',
    'Dockerfile',
    '.dockerignore',
    'demo.py',
    'setup.py',
    'pyproject.toml',
    'config/model.yaml',
    'config/schema.yaml',
}

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent
    if filedir != Path(''):
        filedir.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or (filepath.stat().st_size==0):
        filepath.touch()
    else:
        print(f'file is already presnt at: {filepath}')