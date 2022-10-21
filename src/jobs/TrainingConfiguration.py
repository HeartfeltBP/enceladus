import ast
import random
from configparser import ConfigParser


class TrainingConfiguration():
    def __init__(self):
        self._training_args = dict(
            seed=random.randint(0, 4294967295),
            records_dir=None,
            out_dir=None,
            data_size=None,
            data_split=None,
            steps_per_epoch=None,
            valid_steps=None,
            test_steps=None,
            epochs=None,
            es_patience=None,
            use_multiprocessing=False,
            use_wandb_tracking=False,
            wandb_entity=None,
            wandb_project=None,
        )
        self._model_args = dict(
            batch_size=None,
            learning_rate=None,
            beta_1=None,
            beta_2=None,
            epsilon=None,
            batch_norm=False,
            activation=None,
            dropout_1=None,
            dropout_2=None,
            regularizer=None,
            reg_factor=None,
            init_method=None,
        )

    def get(self, path):
        config = ConfigParser()
        config.read(path)
        config.sections()
        model = self._args(self._model_args, **config['model'])
        training = self._args(self._training_args, **config['training'])
        training = self._get_steps(training, model['batch_size'])
        return training, model

    def _args(self, dic, **kwargs):
        for setting in kwargs:
            try:
                value = kwargs[setting]
                if setting not in ['records_dir', 'out_dir', 'wandb_entity',
                                   'wandb_project', 'activation', 'init_method']:
                    value = ast.literal_eval(value)
                dic[setting] = value
            except KeyError:
                print(f'Invalid argument {setting}')
        return dic

    def _get_steps(self, dic, batch_size):
        for i, item in enumerate(['steps_per_epoch', 'valid_steps', 'test_steps']):
            dic[item] = int((dic['data_size'] * dic['data_split'][i]) / batch_size)
        return dic
