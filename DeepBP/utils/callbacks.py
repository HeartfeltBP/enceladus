import os
import keras
from wandb.keras import WandbCallback
import tensorflow as tf

def get_callbacks(validation_dataset, steps_valid, logger1, args):
    logger1.info('\nSetting up callbacks.')

    tensorboard_dir = args['out_dir'] + 'tensorboard/'
    os.makedirs(tensorboard_dir, exist_ok=True)

    file_writer = tf.summary.create_file_writer(tensorboard_dir)

    # Tensorboard callback
    logger1.info('\nLogging TensorBoard to {}.'.format(tensorboard_dir))

    # Tensorboard callback
    tb_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Early stopping
    es_callback = keras.callbacks.EarlyStopping(monitor='val_mse', patience=args['es_patience'], verbose=1, restore_best_weights=True)
    callbacks = [tb_callback, es_callback]

    if args['use_wandb_tracking']:
        wandb_callback = WandbCallback(
            labels=['sbp', 'dbp'],
            log_weights=True, monitor='val_mse',
            generator=validation_dataset,
            validation_steps=steps_valid,
            log_evaluation=True
        )
        callbacks.append(wandb_callback)
    logger1.info('\nCallbacks are all set up.')
    return callbacks
