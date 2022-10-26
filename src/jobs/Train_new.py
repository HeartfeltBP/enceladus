import os
import keras
import wandb
import tensorflow as tf
from src.models import UNet
from src.utils import RecordsHandler, set_all_seeds, get_strategy, lr_scheduler


class TrainingPipeline():
    def __init__(self, args, config):
        self._args = args
        self._config = config

    def sweep_train(self):
        wandb_config = dict(
            batch_size=32,
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            dropout_1=0.5,
            dropout_2=0.5,
        )

        with wandb.init(
            entity=self._args['wandb_entity'],
            project=self._args['wandb_project'],
            group=str(self._args['wandb_project']),
            sync_tensorboard=True,
            dir=self._args['out_dir'],
            config=wandb_config,
        ):
            dataset, strategy, callbacks = self._setup()
            model = self._initialize_model()
        return

    def _load_dataset(self, seed):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        options = tf.data.Options()
        options.experimental_deterministic = False

        handler = RecordsHandler(data_dir=self._args['records_dir'])
        dataset = handler.read_records(n_cores=10, AUTOTUNE=AUTOTUNE)

        train = dataset['train'].prefetch(AUTOTUNE)
        val = dataset['val'].prefetch(AUTOTUNE)
        test = dataset['test'].prefetch(AUTOTUNE)

        train = train.shuffle(10, seed=seed, reshuffle_each_iteration=True).batch(self._config['batch_size'])
        val = val.shuffle(10, seed=seed, reshuffle_each_iteration=True).batch(self._config['batch_size'])
        test = test.batch(self._config['batch_size'])

        train = train.repeat()
        val = val.repeat()
        return dict(train=train, val=val, test=test)

    def _get_callbacks(self, dataset):
        # Tensorboard
        tensorboard_dir = self._args['out_dir'] + 'tensorboard/'
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
            patience=self._args['es_patience'],
            verbose=1,
            restore_best_weights=True
        )

        # Learning rate decay
        lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # Weights & Biases
        wandb_callback = wandb.WandbCallback(
            monitor='val_loss',
            trainig_data=dataset['train'],
            log_weights=True,
            log_gradients=True,
            generator=dataset['val'],
            validation_steps=self._args['valid_steps'],
            predictions=5,
            input_type='auto',
            output_type='segmentation_mask',
            log_evaluation=True,
        )

        callbacks = [tb_callback, es_callback, lr_callback, wandb_callback]
        return callbacks

    def _setup(self):
        # Set seed for reproducability
        set_all_seeds(self._args['seed'])

        #initialize W&B logging if requested
        wandb_path = self._args['out_dir'] + 'wandb'
        if not os.path.exists(wandb_path):
            os.mkdir(self._args['out_dir'] + 'wandb')
        wandb.tensorboard.patch(root_logdir=self._args['out_dir'])

        strategy = get_strategy()
        dataset = self._load_dataset(seed=self._args['seed'])
        callbacks = self._get_callbacks(dataset)
        return strategy, callbacks

    def _initialize_model(self, strategy, wandb_config):
        with strategy.scope():
            model = UNet(self._config).init()
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

    def _train(self, strategy, wandb_config):
        model = self._initialize_model(strategy, wandb_config)
        model.fit(
            self.dataset['train'],
            validation_data=self.dataset['val'],
            validation_steps=self._args['valid_steps'],
            steps_per_epoch=self._args['steps_per_epoch'],
            epochs=self._args['epochs'],
            callbacks=self.callbacks,
            use_multiprocessing=self._args['use_multiprocessing'],
        )
        return
