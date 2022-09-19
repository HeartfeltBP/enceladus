import os
import gc
import wandb
import tensorflow as tf
from DeepBP.utils import create_model, callbacks
from DeepBP.utils import set_all_seeds, get_logger, get_strategy, get_callbacks
from database_tools.utils import ReadTFRecords


class Train():
    def __init__(self, args):
        self._args = args
        return

    def run(self):
        logger1 = get_logger('output/train.log', 'w')
        logger1.info('Starting training pipeline')

        model, callbacks = self._setup(logger=logger1)
        # model.summary(positions=[.33, .60, .67, 1.])

        dataset = self._load_dataset()

        model = self._train(model, callbacks, dataset, logger1)
        self._save_model(model)

        test_loss, test_acc = self._test(model, dataset['test'], logger1)

        if self._args['use_wandb_tracking']:
            wandb.log({"Test loss":test_loss, "Test accuracy": test_acc})
            wandb.finish()

        # Clean up, free memory (not reliable though)
        tf.keras.backend.clear_session() 
        del model
        gc.collect()

        logger1.info('\nDone.')
        return

    def _get_hyperparameters(self):
        config = {
            'input_shape'   : (625, 1),
            'lr'            : 0.0001,
            'decay'         : 0.0001,
            'kernel_1'      : ( 7),
            'filters_1'     : 64,
            'strides_1'     : (2),
            'max_pooling_1' : (3),
            'kernel_2'      : [(3), (3)],
            'filters_2'     : [64, 64],
            'kernel_3'      : [(3), (3)],
            'filters_3'     : [128, 128],
            'kernel_4'      : [(3), (3)],
            'filters_4'     : [256, 256],
            'kernel_5'      : [(3), (3)],
            'filters_5'     : [512, 512],
            'pooling'       : (2),
            'frame_length'  : 125,
            'frame_step'    : 8,
            'st_units'      : 64,
            'cnn_st_units'  : 64,
            'dropout_1'     : 0.25,
            'dropout_2'     : 0.25,
            'dense_1'       : 32,
            'dense_2'       : 32,
        }
        return config

    def _initialize_model(self, strategy, config):
        with strategy.scope():
            model = create_model(config)
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['lr'], decay=config['decay'])
            model.compile(optimizer=optimizer,
                          loss='mse',
                          metrics=['mae'])
        
        tf.keras.utils.plot_model(
            model,
            to_file=self._args['out_dir'] + 'model.png',
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=300,
        )
        return model

    def _setup(self, logger):
        # Set seed for reproducability
        set_all_seeds(self._args['seed'])

        # Get model config
        model_config = self._get_hyperparameters()

        #initialize W&B logging if requested
        if self._args['use_wandb_tracking']:
            wandb.tensorboard.patch(root_logdir=self._args["out_dir"])
            wandb.init(
                config=model_config,
                entity=self._args["wandb_entity"],
                project=self._args["wandb_project"],
                group=str(self._args["wandb_project"]),
                sync_tensorboard=True,
            )

        # Determine strategy
        strategy = get_strategy(logger)

        # Model setup
        model = self._initialize_model(strategy, model_config)

        # Initialize callbacks
        callbacks = get_callbacks(
            validation_dataset=None,
            steps_valid=None,
            logger1=logger,
            args=self._args,
        )

        return model, callbacks

    def _load_dataset(self, seed=1337):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        options = tf.data.Options()
        options.experimental_deterministic = False

        dataset = ReadTFRecords(data_dir=self._args['tfrecords_dir'],
                                method=self._args['data_method'],
                                n_cores=10,
                                AUTOTUNE=AUTOTUNE).run()
        train = dataset['train'].prefetch(AUTOTUNE)
        val = dataset['val'].prefetch(AUTOTUNE)
        test = dataset['test'].prefetch(AUTOTUNE)

        train = train.shuffle(10, seed=seed, reshuffle_each_iteration=True).batch(self._args['batch_size'])
        val = val.shuffle(10, seed=seed, reshuffle_each_iteration=True).batch(self._args['batch_size'])
        test = test.shuffle(10, seed=seed, reshuffle_each_iteration=True).batch(self._args['batch_size'])

        # Make the data infinite.
        # train = train.repeat()
        # val = val.repeat()
        # test = test.repeat()

        return dict(train=train, val=val, test=test)

    def _train(self, model, callbacks, data, logger):
        logger.info('Starting training.')

        model.fit(
            data['train'],
            validation_data=data['val'],
            validation_steps=self._args['val_steps'],
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

        logger.info('\nFinished evaluation. Loss: {:.4f}, Accuracy: {:.4f}.'.format(test_loss, test_acc))
        return test_loss, test_acc

    def _save_model(self, model):
        model_dir = self._args['out_dir'] + self._args['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
