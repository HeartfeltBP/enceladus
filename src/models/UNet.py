import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout
from keras.regularizers import L1, L2, L1L2

class UNet():
    def __init__(self, config):
        self._config = config
        self._act = self._get_activation(self._config['activation'])
        self._reg = self._get_regularizer(self._config['regularizer'], self._config['reg_factor'])         

    def create_model(self):
        input = Input(shape=(256, 1), name='ppg')
        x1, skip1 = self.contraction_block(input, filters=64, dropout=False)
        x2, skip2 = self.contraction_block(x1, filters=128, dropout=False)
        x3, skip3 = self.contraction_block(x2, filters=256, dropout=False)
        x4, skip4 = self.contraction_block(x3, filters=512, dropout=True)
        x5 = self.bottleneck_block(x4, filters=1024)
        x6 = self.expansion_block(x5, skip4, filters=512)
        x7 = self.expansion_block(x6, skip3, filters=256)
        x8 = self.expansion_block(x7, skip2, filters=128)
        output = self.output_block(x8, skip1, filters=64)
        model = Model(inputs=[input], outputs=[output], name='unet')
        return model

    def basic_block(self, input, filters, kernel_size):
        x = Conv1D(
            filters=filters,
            kernel_size=(kernel_size),
            kernel_regularizer=self._reg,
            padding='same',
        )(input)
        x = BatchNormalization()(x) if self._config['batch_norm'] else x
        x = Activation(self._act)(x)
        return x

    def contraction_block(self, input, filters, dropout):
        x = self.basic_block(input, filters, 3)
        x = self.basic_block(x, filters, 3)
        if dropout:
            x = Dropout(rate=self._config['dropout_1'])(x)
            skip = x
        else:
            skip = x
        x = MaxPooling1D(pool_size=(2))(x)
        return x, skip

    def bottleneck_block(self, input, filters):
        x = self.basic_block(input, filters, 3)
        x = self.basic_block(x, filters, 3)
        x = Dropout(rate=self._config['dropout_2'])(x)
        x = UpSampling1D(size=2)(x)
        return x

    def expansion_block(self, input, skip, filters):
        x = self.basic_block(input, filters, 2)
        x = Concatenate()([x, skip])
        x = self.basic_block(x, filters, 3)
        x = self.basic_block(x, filters, 3)
        x = UpSampling1D(size=2)(x)
        return x

    def output_block(self, input, skip, filters):
        x = self.basic_block(input, filters, 2)
        x = Concatenate()([x, skip])
        x = self.basic_block(x, filters, 3)
        x = self.basic_block(x, filters, 3)
        x = Conv1D(filters=2, kernel_size=(3), kernel_regularizer=self._reg, padding='same')(x)
        output = Conv1D(filters=1, kernel_size=(3), kernel_regularizer=self._reg, padding='same')(x)
        return output

    def _get_activation(self, activation):
        if activation == 'LeakyReLU':
            act = tf.nn.leaky_relu
        elif activation == 'ReLU':
            act = tf.nn.relu
        return act

    def _get_regularizer(self, regularizer, factor):
        if regularizer == 'L1':
            reg = L1(factor)
        elif regularizer == 'L2':
            reg = L2(factor)
        elif regularizer == 'L1L2':
            reg = L1L2(factor)
        else:
            reg = None
        return reg

    def _get_initializer(self, initializer):
        ini = None
        return ini
