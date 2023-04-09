import ast
import wandb
import keras
import tensorflow as tf
from enceladus.models import UNet
from enceladus.utils import set_all_seeds, get_strategy
from database_tools.tools.records import read_records
from database_tools.tools.dataset import ConfigMapper

def get_sweep_values(config):
    out = dict()
    for param, value in config.items():
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


class TrainingPipeline():
    def __init__(self, config_path, no_sweep=False):
        cm = ConfigMapper(config_path)
        self.config = cm.train
        self.model_config = cm.model

        self.sweep_config = get_sweep_values(cm.sweep.__dict__)
        self.no_sweep = no_sweep
        if no_sweep:
            self.default_config = {}
            for item, value in self.sweep_config['parameters'].items():
                self.default_config[item] = value['values']
        else:
            self.default_config = dict(
                batch_size=32,
                learning_rate=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                dropout_1=0.5,
                dropout_2=0.5,
		)

    def run(self):
        set_all_seeds(self.config.seed)
        self.strategy = get_strategy(self.config.hardware)

        if not self.no_sweep:
            sweep_id = wandb.sweep(
            self.sweep_config,
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
		    )
            wandb.agent(sweep_id, function=self._train)
        else:
            self._train()
        wandb.finish()
        return

    def _load_dataset(self, wandb_config):
        epochs = self.config.epochs
        batch_size = wandb_config.batch_size

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        if self.config.inputs == 'ppg/vpg/apg':
            dataset = read_records(self.config.records_path, n_cores=self.config.n_cores)

        train = dataset['train'].prefetch(AUTOTUNE).shuffle(10*batch_size).batch(batch_size, num_parallel_calls=AUTOTUNE)
        val = dataset['val'].prefetch(AUTOTUNE).shuffle(10*batch_size).batch(batch_size, num_parallel_calls=AUTOTUNE)
        if self.config.hardware in ['Pegasus']:
            train = train.cache()
            val = val.cache()
        train = train.repeat(epochs)
        val = val.repeat(epochs)
        return dict(train=train, val=val)

    def _get_callbacks(self, dataset, valid_steps):
        # Early stopping
        es_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.es_patience,
            min_delta=self.config.es_min_delta,
            restore_best_weights=True,
            verbose=1,
        )

        # Learning rate decay
        lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.lr_decay_factor,
            patience=self.config.lr_patience,
            min_delta=self.config.lr_min_delta,
            mode='min',
            verbose=1,
        )

        # Weights & Biases
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_loss',
            mode='min',
            validation_data=dataset['val'],
            validation_steps=valid_steps,
            save_model=self.config.save_model,
        )

        callbacks = [lr_callback, wandb_callback]
        return callbacks

    def _optimizer(self, name, learning_rate, beta_1, beta_2, epsilon):
        if name == 'Adam':
            opt = tf.keras.optimizers.Adam
        elif name == 'Nadam':
            opt = tf.keras.optimizers.Nadam
        elif name == 'Adamax':
            opt = tf.keras.optimizers.Adamax
        return opt(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            decay=0.00001,
        )

    def _train(self):
        run = wandb.init(
            project=self.config.wandb_project,
            config=self.default_config,
        )
        self.model_config.set_attr({'dropout': wandb.config.dropout})

        steps_per_epoch = int((self.config.data_size * self.config.data_split[0]) / wandb.config.batch_size)
        valid_steps = int((self.config.data_size * self.config.data_split[1]) / wandb.config.batch_size)

        dataset = self._load_dataset(wandb.config)
        callbacks = self._get_callbacks(dataset, valid_steps)

        strategy = tf.distribute.OneDeviceStrategy(device='/device:GPU:0')
        with strategy.scope():
            if self.config.inputs == 'ppg/vpg/apg':
                model = UNet(self.model_config).init()
            else:
                raise ValueError(f'Invalid input configuration')

            optimizer = self._optimizer(
                name=self.config.optimizer,
                learning_rate=wandb.config.learning_rate,
                beta_1=wandb.config.beta_1,
                beta_2=wandb.config.beta_2,
                epsilon=wandb.config.epsilon,
            )

            model.compile(
                optimizer=optimizer,
                loss=self.config.loss,
                metrics=['mae'],
            )

        print('Fitting...')
        run = model.fit(
            dataset['train'],
            validation_data=dataset['val'],
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=valid_steps,
            callbacks=callbacks,
            use_multiprocessing=True,
        )
        if self.no_sweep:
            model.save('enceladus_model')
        return



