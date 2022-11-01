import os
import wandb
import keras
import tensorflow as tf
from Enceladus.models import UNet
from Enceladus.utils import RecordsHandler, set_all_seeds, get_strategy, lr_scheduler


class TrainingPipeline():
    def __init__(self, config, model_config, sweep_config, no_sweep=False):
        self.config = config
        self.model_config = model_config
        self.sweep_config = sweep_config
        self.no_sweep = no_sweep
        if no_sweep:
            self.default_config = sweep_config
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
        set_all_seeds(self.config['seed'])
        self.strategy = get_strategy()

        if not self.no_sweep:
            sweep_id = wandb.sweep(
            self.sweep_config,
            entity=self.config['wandb_entity'],
            project=self.config['wandb_project'],
		    )
            wandb.agent(sweep_id, function=self._train)
        else:
            self._train()
        return

    def _load_dataset(self, wandb_config):
        epochs = self.config['epochs']
        batch_size = wandb_config.batch_size

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        handler = RecordsHandler(data_dir=self.config['records_dir'])
        dataset = handler.read_records(n_cores=self.config['n_cores'], AUTOTUNE=AUTOTUNE)

        train = dataset['train'].prefetch(AUTOTUNE).shuffle(10*batch_size).batch(batch_size).repeat(epochs)
        val = dataset['val'].prefetch(AUTOTUNE).shuffle(10*batch_size).batch(batch_size).repeat(epochs)
        return dict(train=train, val=val)

    def _get_callbacks(self):
        # Early stopping
        es_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['es_patience'],
            restore_best_weights=True
        )

        # Learning rate decay
        lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
        )

        # Weights & Biases
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_loss',
						mode='min',
        )

        callbacks = [es_callback, lr_callback, wandb_callback]
        return callbacks

    def _optimizer(self, name, learning_rate, beta_1, beta_2, epsilon):
        if name == 'Adam':
            opt = tf.keras.optimizers.Adam
        elif name == 'Nadam':
            opt = tf.keras.optimizers.Nadam
        return opt(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
        )

    def _train(self):
        wandb.init(config=self.default_config)
        self.model_config['dropout_1'] = wandb.config.dropout_1
        self.model_config['dropout_2'] = wandb.config.dropout_2
        with self.strategy.scope():
            model = UNet(self.model_config).init()

            optimizer = self._optimizer(
                name=self.config['optimizer'],
                learning_rate=wandb.config.learning_rate,
                beta_1=wandb.config.beta_1,
                beta_2=wandb.config.beta_2,
                epsilon=wandb.config.epsilon,
            )

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
            )
        dataset = self._load_dataset(wandb.config)
        callbacks = self._get_callbacks()

        print('Fitting...')
        run = model.fit(
            dataset['train'],
            validation_data=dataset['val'],
            epochs=self.config['epochs'],
            callbacks=callbacks,
            use_multiprocessing=True,
        )
        return
