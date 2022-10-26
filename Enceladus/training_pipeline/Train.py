import os
import wandb
import keras
import tensorflow as tf
from Enceladus.models import UNet
from Enceladus.utils import RecordsHandler, set_all_seeds, get_strategy, lr_scheduler


class TrainingPipeline():
    def __init__(self, config, model_config, sweep_config):
        self.config = config
        self.model_config = model_config
        self.sweep_config = sweep_config

    def run(self):
        set_all_seeds(self.config['seed'])
        self.strategy = get_strategy()
        sweep_id = wandb.sweep(
            self.sweep_config,
            entity=self.config['wandb_entity'],
            project=self.config['wandb_project'],
        )
        wandb.agent(sweep_id, function=self._train, count=10)
        return

    def _load_dataset(self, wandb_config):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        options = tf.data.Options()
        options.experimental_deterministic = False

        handler = RecordsHandler(data_dir=self.config['records_dir'])
        dataset = handler.read_records(n_cores=10, AUTOTUNE=AUTOTUNE)

        train = dataset['train'].prefetch(AUTOTUNE)
        val = dataset['val'].prefetch(AUTOTUNE)
        test = dataset['test'].prefetch(AUTOTUNE)

        train = train.shuffle(10).batch(wandb_config.batch_size)
        val = val.shuffle(10).batch(wandb_config.batch_size)
        test = test.batch(wandb_config.batch_size)

        train = train.repeat()
        val = val.repeat()
        return dict(train=train, val=val, test=test)

    def _get_callbacks(self, dataset, valid_steps):
        # Tensorboard
        tensorboard_dir = self.config['out_dir'] + 'tensorboard/'
        os.makedirs(tensorboard_dir, exist_ok=True)
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )

        # Early stopping
        es_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['es_patience'],
            verbose=1,
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
            generator=dataset['val'],
            validation_steps=valid_steps,
            log_evaluation=True,
            log_weights=True,
        )

        callbacks = [tb_callback, es_callback, lr_callback]
        return callbacks

    def _train(self):
        default_config = dict(
            batch_size=32,
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            dropout_1=0.5,
            dropout_2=0.5,
        )
        wandb.tensorboard.patch(root_logdir=self.config['out_dir'])
        wandb.init(sync_tensorboard=True, config=default_config, group=str(self.config['wandb_project']))
        self.model_config['dropout_1'] = wandb.config.dropout_1
        self.model_config['dropout_2'] = wandb.config.dropout_2
        with self.strategy.scope():
            model = UNet(self.model_config).init()
            optimizer = tf.keras.optimizers.Adam(
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

        steps_per_epoch = int((self.config['data_size'] * self.config['data_split'][0]) / wandb.config.batch_size)
        valid_steps = int((self.config['data_size'] * self.config['data_split'][1]) / wandb.config.batch_size)

        callbacks = self._get_callbacks(dataset, valid_steps)

        print('Fitting...')
        model.fit(
            dataset['train'],
            validation_data=dataset['val'],
            validation_steps=valid_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=1,
        )
        return
