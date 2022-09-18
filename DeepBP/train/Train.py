import tensorflow as tf
from DeepBP.utils import create_model, callbacks
from DeepBP.utils import get_logger, get_strategy, get_callbacks


class Train():
    def __init__(self, args):
        """
        args = {
            train,
            val,
            test,
        }
        """
        self._args = args
        return

    def run(self):
        logger1 = get_logger('train.log', 'w')
        logger1.info('Starting training pipeline')

        model, callbacks = self._setup(logger=logger1, args=self._args)
        return

    def _get_hyperparameters(self):
        return

    def _configure_model(self):
        return

    def _train(self):
        return

    def _test(self):
        return

    def _setup(self, logger, args):
        strategy = get_strategy(logger)

        model = create_model()
        callbacks = get_callbacks()
        return model, callbacks

    def _save_model(self):
        return