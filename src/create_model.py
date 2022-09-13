import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, \
                         AveragePooling1D, Add, GRU, Dense, Permute, Reshape, Dropout
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

def create_model(input_shape=(625, 1), lr=0.0001, plot=False):
    input = Input(shape=input_shape)
    output = activebp(input, lr=lr)
    model = Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, decay=0.0001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])

    if plot:
        plot_model(model=model, to_file='model.png', show_shapes=True)
    return model

def activebp(x, lr):
    x1 = resnet50(x)
    x1 = GRU(65)(x1)
    x1 = BatchNormalization()(x1)

    x2 = spectro_temporal_block(x)

    output = Add()([x1, x2])
    output = Dense(32, activation='relu', kernel_regularizer=l2(lr))(output)
    output = Dropout(0.25)(output)
    output = Dense(32, activation='relu', kernel_regularizer=l2(lr))(output)
    output = Dropout(0.25)(output)
    output = Dense(2, activation='relu')(output)
    return output

def resnet50(x):
    x = Conv1D(kernel_size=(7), filters=64, strides=(2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D((3), strides=(2))(x)

    x = convolutional_block(x, kernel_sizes=[(1), (3), (1)], filters=[64, 64, 256])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[64, 64, 256])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[64, 64, 256])

    x = convolutional_block(x, kernel_sizes=[(1), (3), (1)], filters=[128, 128, 512])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[128, 128, 512])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[128, 128, 512])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[128, 128, 512])

    x = convolutional_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[256, 256, 1024])

    x = convolutional_block(x, kernel_sizes=[(1), (3), (1)], filters=[512, 512, 2048])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[512, 512, 2048])
    x = identity_block(x, kernel_sizes=[(1), (3), (1)], filters=[512, 512, 2048])

    x = AveragePooling1D(pool_size=(2), padding='same')(x)
    return x

def convolutional_block(x, kernel_sizes, filters):
    kernel_size0 = kernel_sizes[0]
    kernel_size1 = kernel_sizes[1]
    kernel_size2 = kernel_sizes[2]
    filter0 = filters[0]
    filter1 = filters[1]
    filter2 = filters[2]

    x_skip = x

    x = Conv1D(kernel_size=kernel_size0, filters=filter0, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size2, filters=filter2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_skip = Conv1D(kernel_size=kernel_size2, filters=filter2, padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def identity_block(x, kernel_sizes, filters):
    kernel_size0 = kernel_sizes[0]
    kernel_size1 = kernel_sizes[1]
    kernel_size2 = kernel_sizes[2]
    filter0 = filters[0]
    filter1 = filters[1]
    filter2 = filters[2]

    x_skip = x

    x = Conv1D(kernel_size=kernel_size0, filters=filter0, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size2, filters=filter2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def spectro_temporal_block(x):
    frame_length = 125
    frame_step = 8

    x = Permute((2, 1))(x)
    x = tf.signal.stft(x,
                       frame_length=frame_length,
                       frame_step=frame_step)
    x = tf.abs(x)
    x = Reshape((63, 65))(x)
    x = GRU(65)(x)
    x = BatchNormalization()(x)
    return x
