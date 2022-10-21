import os
import gc
import sys
import wandb
import tensorflow as tf
from src.models import UNet
from src.utils import RecordsHandler, set_all_seeds, get_logger, get_strategy, get_callbacks


class Train():
    def __init__(self, args, config):
        self._args = args
        self._config = config

    def run(self):
        logger = get_logger('output/train.log', 'w')
        logger.info('Starting training pipeline')

        model, callbacks, dataset = self._setup(logger=logger)

        model = self._train(model, callbacks, dataset, logger)

        test_loss, test_acc = self._test(model, dataset['test'], logger)

        if self._args['use_wandb_tracking']:
            wandb.log({"Test loss":test_loss, "Test accuracy": test_acc})
            wandb.finish()

        # Clean up, free memory (not reliable though)
        tf.keras.backend.clear_session() 
        del model
        gc.collect()

        logger.info('\nDone.')
        return

    def _setup(self, logger):
        # Set seed for reproducability
        set_all_seeds(self._args['seed'])

        #initialize W&B logging if requested
        if self._args['use_wandb_tracking']:
            wandb_path = self._args['out_dir'] + 'wandb'
            if not os.path.exists(wandb_path):
                os.mkdir(self._args['out_dir'] + 'wandb')
            wandb.tensorboard.patch(root_logdir=self._args['out_dir'])
            wandb.init(
                entity=self._args['wandb_entity'],
                project=self._args['wandb_project'],
                group=str(self._args['wandb_project']),
                sync_tensorboard=True,
                dir=self._args['out_dir'],
            )

        # Determine strategy
        strategy = get_strategy(logger)

        # Model setup
        model = self._initialize_model(strategy)

        # Load data
        dataset = self._load_dataset()

        # Initialize callbacks
        callbacks = get_callbacks(
            validation_dataset=dataset['val'],
            logger1=logger,
            args=self._args,
        )
        return model, callbacks, dataset

    def _initialize_model(self, strategy):
        with strategy.scope():
            model = UNet(self._config).create_model()
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self._config['learning_rate'],
                beta_1=self._config['beta_1'],
                beta_2=self._config['beta_2'],
                epsilon=self._config['epsilon'],
            )
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
            )
        return model

    def _load_dataset(self, seed=1337):
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

    def _train(self, model, callbacks, data, logger):
        logger.info('Starting training.')

        model.fit(
            data['train'],
            validation_data=data['val'],
            validation_steps=self._args['valid_steps'],
            steps_per_epoch=self._args['steps_per_epoch'],
            epochs=self._args['epochs'],
            callbacks=callbacks,
            use_multiprocessing=self._args['use_multiprocessing'],
        )

        logger.info('Training finished.')
        return model

    def _test(self, model, test_data, logger):
        logger.info('\nStarting evaluation on test data.')

        test_loss, test_acc = model.evaluate(x=test_data, steps=self._args['test_steps'])

        logger.info('\nFinished evaluation. Loss (MSE): {:.4f}, MAE: {:.4f}.'.format(test_loss, test_acc))
        return test_loss, test_acc

    def _save_model(self, model):
        model_dir = self._args['out_dir'] + 'model/'
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
