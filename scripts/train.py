import os
from enceladus.workflows import TrainingPipeline
from enceladus.utils import GetConfiguration

repo_dir = os.getcwd().split('notebooks')[0]
os.chdir(repo_dir)

pipeline, model, sweep = GetConfiguration().run('config/20230220.ini')
worker = TrainingPipeline(
    config=pipeline,
    model_config=model,
    sweep_config=sweep,
    no_sweep=True,
)
worker.run()
