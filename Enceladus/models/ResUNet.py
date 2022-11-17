import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout, Add
from keras.initializers.initializers_v2 import GlorotUniform, HeUniform
from keras.regularizers import L1, L2, L1L2


class ResUNet():
    def __init__(self, config):
        self.config = config
        ini, act, reg = self.get_model_components(self.config)
        self.ini = ini
        self.act = act
        self.reg = reg

    def init(self):
        input = Input(shape=(256, 1), name='ppg')
        x1, skip1 = self.contraction_block(input, filters=64, pooling=True)
        x2, skip2 = self.contraction_block(x1, filters=128, pooling=True)
        x3, skip3 = self.contraction_block(x2, filters=256, pooling=True)
        x4, skip4 = self.contraction_block(x3, filters=512, pooling=True)
        x5 = self.contraction_block(x4, filters=1024, pooling=False)
        x5 = Dropout(self.config['dropout_1'])(x5)
        skip5 = x5
        x5 = MaxPooling1D(pool_size=(2))(x5)

        x6 = self.contraction_block(x5, filters=2048, pooling=False)
        x6 = Dropout(self.config['dropout_2'])(x6)
        x6 = UpSampling1D(size=2)(x6)

        x7 = self.expansion_block(x6, skip5, filters=1024, sampling=True)
        x8 = self.expansion_block(x7, skip4, filters=512, sampling=True)
        x9 = self.expansion_block(x8, skip3, filters=256, sampling=True)
        x10 = self.expansion_block(x9, skip2, filters=128, sampling=True)
        x11 = self.expansion_block(x10, skip1, filters=64, sampling=False)
        output = self.output_block(x11)
        model = Model(inputs=[input], outputs=[output], name='unet')
        return model

    def get_model_components(self, config):
        initializers = dict(
            GlorotUniform=GlorotUniform(),
            HeUniform=HeUniform(),
        )
        activations = dict(
            ReLU=tf.nn.relu,
            LeakyReLU=tf.nn.leaky_relu,
        )
        regularizers = dict(
            L1=L1(config['reg_factor_1']),
            L2=L2(config['reg_factor_1']),
            L1L2=L1L2(config['reg_factor_1'], config['reg_factor_2']),
        )
        if config['initializer'] != 'None':
            ini = initializers[config['initializer']]
        else:
            ini = None
        if config['activation'] != 'None':
            act = activations[config['activation']]
        else:
            act = None
        if config['regularizer'] != 'None':
            reg = regularizers[config['regularizer']]
        else:
            reg = None
        return ini, act, reg

    def basic_block(self, input, filters, size):
        x = Conv1D(
            filters=filters,
            kernel_size=(size),
            kernel_initializer=self.ini,
            kernel_regularizer=self.reg,
            padding='same',
        )(input)
        x = BatchNormalization()(x) if self.config['batch_norm'] else x
        x = Activation(self.act)(x)
        return x

    def contraction_block(self, input, filters, pooling):
        x = self.basic_block(input, filters, 3)
        res_skip = x
        x = self.basic_block(x, filters, 3)
        x = Add()([x, res_skip])
        if pooling:
            skip = x
            x = MaxPooling1D(pool_size=(2))(x)
            return x, skip
        else:
            return x

    def expansion_block(self, input, skip, filters, sampling):
        x = self.basic_block(input, filters, 2)
        x = Concatenate()([x, skip])
        x = self.basic_block(x, filters, 3)
        res_skip = x
        x = self.basic_block(x, filters, 3)
        x = Add()([x, res_skip])
        x = UpSampling1D(size=2)(x) if sampling else x
        return x

    def output_block(self, input):
        x = Conv1D(
            filters=2,
            kernel_size=(3),
            kernel_initializer=self.ini,
            kernel_regularizer=self.reg,
            padding='same',
        )(input)
        x = Activation(self.act)(x)
        x = Conv1D(
            filters=1,
            kernel_size=(3),
            kernel_initializer=self.ini,
            kernel_regularizer=self.reg,
            padding='same'
        )(x)
        x = Activation('linear')(x)
        return x
