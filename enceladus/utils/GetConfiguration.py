import ast
from configparser import ConfigParser


class GetConfiguration():

    def run(self, path):
        config = ConfigParser()
        config.read(path)
        config.sections()

        pipeline_literals = [
            'epochs',
            'es_patience',
            'es_min_delta',
            'lr_decay_factor',
            'lr_patience',
            'lr_min_delta',
            'n_cores',
            'seed',
            'data_size',
            'data_split',
            'save_model',
        ]
        pipeline = self.get_values(config['pipeline'], pipeline_literals)

        model_literals = []
        model = self.get_values(config['model'], model_literals)

        sweep_literals = [
            'batch_size',
            'learning_rate',
            'beta_1',
            'beta_2',
            'epsilon',
            'dropout'
        ]
        sweep = self.get_sweep_values(config['sweep'], sweep_literals, pipeline)
        return pipeline, model, sweep

    def get_values(self, config, literals):
        out = dict()
        for param, value in config.items():
            if param in literals:
                value = ast.literal_eval(value)
            out[param] = value
        return out

    def get_sweep_values(self, config, literals, pipeline):
        out = dict()
        for param, value in config.items():
            if param in literals:
                value = ast.literal_eval(value)
            out[param] = dict(values=value)

        out = dict(
            method='bayes',
            metric=dict(
                name='val_loss',
                goal='minimize',
            ),
            parameters=out
        )
        return out
