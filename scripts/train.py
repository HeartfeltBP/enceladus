import os
from enceladus.workflows import TrainingPipeline

repo_dir = os.getcwd().split('notebooks')[0]
os.chdir(repo_dir)

worker = TrainingPipeline(
    config_path='/home/cam/Documents/database_tools/data/mimic3-data-20230408/config.ini',
    no_sweep=True,
)
worker.run()
