import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout
from keras.initializers.initializers_v2 import GlorotUniform, HeUniform
from keras.regularizers import L1, L2, L1L2


class UNet():
    def __init__(self, config):
        self._config = config
        self._ini, self._act, self._reg = self.get_config_components()

    def init(self):
        input = Input(shape=(256, 1), name='ppg')
        x1, skip1 = self.contraction_block(input, filters=64, pooling=True)
        x2, skip2 = self.contraction_block(x1, filters=128, pooling=True)
        x3, skip3 = self.contraction_block(x2, filters=256, pooling=True)

        x4 = self.contraction_block(x3, filters=512, pooling=False)
        x4 = Dropout(self._config['dropout_1'])(x4)
        skip4 = x4
        x4 = MaxPooling1D(pool_size=(2))(x4)

        x5 = self.contraction_block(x4, filters=1024, pooling=False)
        x5 = Dropout(self._config['dropout_2'])(x5)
        x5 = UpSampling1D(size=2)(x5)

        x6 = self.expansion_block(x5, skip4, filters=512, sampling=True)
        x7 = self.expansion_block(x6, skip3, filters=256, sampling=True)
        x8 = self.expansion_block(x7, skip2, filters=128, sampling=True)
        x9 = self.expansion_block(x8, skip1, filters=64, sampling=False)
        output = self.output_block(x9)
        model = Model(inputs=[input], outputs=[output], name='unet')
        return model

    def basic_block(self, input, filters, size):
            x = Conv1D(
                filters=filters,
                kernel_size=(size),
                kernel_initializer=self._ini,
                kernel_regularizer=self._reg,
                padding='same',
            )(input)
            x = BatchNormalization()(x) if self._config['batch_norm'] else x
            x = Activation(self._act)(x)
            return x

    def contraction_block(self, input, filters, pooling):
        x = self.basic_block(input, filters, 3)
        x = self.basic_block(x, filters, 3)
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
        x = self.basic_block(x, filters, 3)
        x = UpSampling1D(size=2)(x) if sampling else x
        return x

    def output_block(self, input):
        x = Conv1D(
            filters=2,
            kernel_size=(3),
            kernel_initializer=self._ini,
            kernel_regularizer=self._reg,
            padding='same',
        )(input)
        x = Conv1D(
            filters=1,
            kernel_size=(3),
            kernel_initializer=self._ini,
            kernel_regularizer=self._reg,
            padding='same'
        )(x)
        return x

    def get_config_components(self):
        initializers = dict(
            GlorotUniform=GlorotUniform(),
            HeUniform=HeUniform(),
        )
        activations = dict(
            ReLU=tf.nn.relu,
            LeakyReLU=tf.nn.leaky_relu,
        )
        regularizers = dict(
            L1=L1(self._config['reg_factor_1']),
            L2=L2(self._config['reg_factor_1']),
            L1L2=L1L2(self._config['reg_factor_1'], self._config['reg_factor_2']),
            none=None,
        )
        ini = initializers[self._config['initializer']]
        act = activations[self._config['activation']]
        reg = regularizers[self._config['regularizer']]
        return ini, act, reg



