import os
import random
import numpy as np
import tensorflow as tf

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

def get_strategy(hardware):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if hardware == 'bongo':
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif hardware == 'Pegasus':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise ValueError(f'Invalid hardware option {hardware}')
    return strategy

def lr_scheduler(epoch, lr):
    if epoch < 7:
      return lr
    else:
      return lr * tf.math.exp(-0.5)
