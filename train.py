from Enceladus.jobs import TrainingPipeline
from Enceladus.utils import GetConfiguration

pipeline, model, sweep = GetConfiguration().run('config-sweep.ini')
worker = TrainingPipeline(
    config=pipeline,
    model_config=model,
    sweep_config=sweep,
    no_sweep=False,
)
worker.run()
