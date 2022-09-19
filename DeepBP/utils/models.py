import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, \
                         AveragePooling1D, Add, Bidirectional, LSTM, Dense, Permute, Reshape, Dropout, Flatten
from keras.regularizers import l2

def create_model(config):
    if config['model'] == 'deepbp':
        input = Input(shape=config['input_shape'])
        output = deepbp(input, config)
        model = Model(inputs=[input], outputs=[output], name='deepbp')
    if config['model'] == 'deepbp_2':
        input = Input(shape=config['input_shape'])
        output = deepbp_2(input, config)
        model = Model(inputs=[input], outputs=[output], name='deepbp')
    return model

def deepbp_2(x, config):
    x = resnet(x, config)

    output = Dense(config['dense_1'], activation='relu', kernel_regularizer=l2(config['lr']))(x)
    output = Dropout(config['dropout_1'])(output)
    output = Dense(config['dense_2'], activation='relu', kernel_regularizer=l2(config['lr']))(output)
    output = Dropout(config['dropout_1'])(output)
    output = Dense(2, activation='relu')(output)
    return output

def deepbp(x, config):
    x1 = resnet(x, config)

    # CNN LSTM Layer
    x1 = Bidirectional(LSTM(config['cnn_st_units'],
                            activation='tanh',
                            recurrent_activation='sigmoid'))(x1)
    x1 = BatchNormalization()(x1)

    # Spectro-temporal Layer
    x2 = spectro_temporal_block(x, config)

    # Output layer
    output = Add()([x1, x2])
    output = Dense(config['dense_1'], activation='relu', kernel_regularizer=l2(config['lr']))(output)
    output = Dropout(config['dropout_1'])(output)
    output = Dense(config['dense_2'], activation='relu', kernel_regularizer=l2(config['lr']))(output)
    output = Dropout(config['dropout_1'])(output)
    output = Dense(2, activation='relu')(output)
    return output

def resnet(x, config):
    # Layer 1
    x = Conv1D(kernel_size=config['kernel_1'], filters=config['filters_1'], strides=config['stride_1'], padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(config['max_pooling_1'], strides=config['stride_1'])(x)

    # Layer 2
    x = identity_block(x, kernel_sizes=config['kernel_2'], filters=config['filters_2'])
    x = identity_block(x, kernel_sizes=config['kernel_2'], filters=config['filters_2'])
    x = identity_block(x, kernel_sizes=config['kernel_2'], filters=config['filters_2'])

    # Layer 3
    x = convolutional_block(x, kernel_sizes=config['kernel_3'], filters=config['filters_3'], stride=config['stride_3'])
    x = identity_block(x, kernel_sizes=config['kernel_3'], filters=config['filters_3'])
    x = identity_block(x, kernel_sizes=config['kernel_3'], filters=config['filters_3'])
    x = identity_block(x, kernel_sizes=config['kernel_3'], filters=config['filters_3'])

    # Layer 4
    x = convolutional_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'], stride=config['stride_4'])
    x = identity_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'])
    x = identity_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'])
    x = identity_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'])
    x = identity_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'])
    x = identity_block(x, kernel_sizes=config['kernel_4'], filters=config['filters_4'])

    # Layer 5
    x = convolutional_block(x, kernel_sizes=config['kernel_5'], filters=config['filters_5'], stride=config['stride_5'])
    x = identity_block(x, kernel_sizes=config['kernel_5'], filters=config['filters_5'])
    x = identity_block(x, kernel_sizes=config['kernel_5'], filters=config['filters_5'])

    # Pooling
    x = AveragePooling1D(pool_size=config['pooling'], padding='same')(x)
    x = Flatten()(x)
    x = Dense(config['dense_0'], activation='relu')(x)
    return x

def convolutional_block(x, kernel_sizes, filters, stride):
    kernel_size0 = kernel_sizes[0]
    kernel_size1 = kernel_sizes[1]
    filter0 = filters[0]
    filter1 = filters[1]

    x_skip = x

    x = Conv1D(kernel_size=kernel_size0, filters=filter0, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(kernel_size=kernel_size1, filters=filter1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_skip = Conv1D(kernel_size=kernel_size1, filters=filter1, strides=stride, padding='same')(x_skip)
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

def spectro_temporal_block(x, config):
    x = Permute((2, 1))(x)
    x = tf.signal.stft(x,
                       frame_length=config['frame_length'],
                       frame_step=config['frame_step'])
    x = tf.abs(x)
    x = Reshape((63, 65))(x)
    x = Bidirectional(LSTM(config['st_units'],
                           activation='tanh',
                           recurrent_activation='sigmoid',
                           name='spectro_temperal_layer'))(x)
    x = BatchNormalization()(x)
    return x
