import os
import keras
import random
import logging
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

def get_logger(filename, filemode):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=filename,
        filemode=filemode,
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger1 = logging.getLogger("Standard")
    return logger1

def get_strategy(logger1, xla=0, fp16=0, no_cuda=0):
    '''
    Determines the strategy under which the network is trained.
  
    From https://github.com/huggingface/transformers/blob/8eb7f26d5d9ce42eb88be6f0150b22a41d76a93d/src/transformers/training_args_tf.py
  
    returns the strategy object
  
    '''
    logger1.info("TensorFlow: setting up strategy")

    # To get rid of flag warnings.
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # try:
    #     os.system('for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done')
    # except:
    #     pass

    if xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices("GPU")
    # Set to float16 at first
    if fp16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if no_cuda:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            tpu = None
  
    if tpu:
        # Set to bfloat16 in case of TPU
        if fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
            tf.keras.mixed_precision.experimental.set_policy(policy)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
    
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
    elif len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif len(gpus) > 1:
        # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise ValueError("Cannot find the proper strategy! Please check your environment properties.")

    logger1.info(f"Using strategy: {strategy}")
    return strategy

def get_callbacks(validation_dataset, logger1, args):
    logger1.info('\nSetting up callbacks.')

    tensorboard_dir = args['out_dir'] + 'tensorboard/'
    os.makedirs(tensorboard_dir, exist_ok=True)

    file_writer = tf.summary.create_file_writer(tensorboard_dir)

    # Tensorboard callback
    logger1.info('\nLogging TensorBoard to {}.'.format(tensorboard_dir))

    # Tensorboard callback
    tb_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Early stopping
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args['es_patience'], verbose=1, restore_best_weights=True)
    callbacks = [tb_callback, es_callback]

    if args['use_wandb_tracking']:
        wandb_callback = WandbCallback(
            # training_data=training_dataset,
            # log_gradients=True,
            log_weights=True,
            monitor='val_loss',
            generator=validation_dataset,
            validation_steps=args['valid_steps'],
            log_evaluation=True
        )
        callbacks.append(wandb_callback)
    logger1.info('\nCallbacks are all set up.')
    return callbacks
