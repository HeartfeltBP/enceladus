from src.models import UNet
from src.utils import RecordsHandler, set_all_seeds, get_logger, get_strategy, get_callbacks


class TrainingPipeline():
    def __init__(self, args, config):
        self._args = args
        self._config = config

    def run(self):
        return
    
    def _get_strategy(self):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        return strategy

    def _initialize_model(self, strategy):
        with strategy.scope():
            model = UNet().init()
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=None,
                beta_1=None,
                beta_2=None,
                epsilon=None,
            )
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
            )
        return model

    def _setup(self):
        # Set seed for reproducability
        set_all_seeds(self._args['seed'])

        strategy = self._get_strategy()
        model = self._initialize_model(strategy)
        return




import tensorflow as tf

