import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, \
                         AveragePooling1D, Add, Bidirectional, LSTM, Dense, Permute, Reshape, Dropout
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

def create_model(input_shape=(625, 1), lr=0.0001, decay=0.0001, plot=False):
    input = Input(shape=input_shape)
    output = activebp(input, lr=lr)
    model = Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, decay=decay)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])

    if plot:
        plot_model(model=model, to_file='model.png', show_shapes=True)
    return model

def activebp(x, lr):
    x1 = resnet(x)
    x1 = Bidirectional(LSTM(64,
                            activation='tanh',
                            recurrent_activation='sigmoid'))(x1)
    x1 = BatchNormalization()(x1)

    x2 = spectro_temporal_block(x)

    output = Add()([x1, x2])
    output = Dense(32, activation='relu', kernel_regularizer=l2(lr))(output)
    output = Dropout(0.25)(output)
    output = Dense(32, activation='relu', kernel_regularizer=l2(lr))(output)
    output = Dropout(0.25)(output)
    output = Dense(2, activation='relu')(output)
    return output

def resnet(x):
    x = Conv1D(kernel_size=(7), filters=64, strides=(2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D((3), strides=(2))(x)

    x = convolutional_block(x, kernel_sizes=[(3), (3)], filters=[64, 64])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[64, 64])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[64, 64])

    x = convolutional_block(x, kernel_sizes=[(3), (3)], filters=[128, 128])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[128, 128])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[128, 128])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[128, 128])

    x = convolutional_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[256, 256])

    x = convolutional_block(x, kernel_sizes=[(3), (3)], filters=[512, 512])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[512, 512])
    x = identity_block(x, kernel_sizes=[(3), (3)], filters=[512, 512])

    x = AveragePooling1D(pool_size=(2), padding='same')(x)
    return x

def convolutional_block(x, kernel_sizes, filters):
    kernel_size0 = kernel_sizes[0]
    kernel_size1 = kernel_sizes[1]
    filter0 = filters[0]
    filter1 = filters[1]

    x_skip = x

    x = Conv1D(kernel_size=kernel_size0, filters=filter0, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_skip = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def identity_block(x, kernel_sizes, filters):
    kernel_size0 = kernel_sizes[0]
    kernel_size1 = kernel_sizes[1]
    filter0 = filters[0]
    filter1 = filters[1]

    x_skip = x

    x = Conv1D(kernel_size=kernel_size0, filters=filter0, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x)
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
    x = Bidirectional(LSTM(64,
                           activation='tanh',
                           recurrent_activation='sigmoid'))(x)
    x = BatchNormalization()(x)
    return x
