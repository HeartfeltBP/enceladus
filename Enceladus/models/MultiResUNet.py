import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout, Add
from keras.initializers.initializers_v2 import GlorotUniform, HeUniform
from keras.regularizers import L1, L2, L1L2


class MultiResUNet():
    def __init__(self, config):
        self.config = config
        ini, act, reg = self.get_model_components(self.config)
        self.ini = ini
        self.act = act
        self.reg = reg

    def init(self):
        ppg = Input(shape=(256, 1), name='ppg')
        vpg = Input(shape=(256, 1), name='vpg')
        apg = Input(shape=(256, 1), name='apg')

        apg_1, apg_skip_1 = self.contraction_block(apg, 32)
        vpg_1, vpg_skip_1 = self.contraction_block(vpg, 32)
        ppg_1, ppg_skip_1 = self.contraction_block(ppg, 32)

        apg_2, apg_skip_2 = self.contraction_block(apg_1, 64)
        vpg_2, vpg_skip_2 = self.contraction_block(vpg_1, 64)
        ppg_2, ppg_skip_2 = self.contraction_block(ppg_1, 64)

        apg_3, apg_skip_3 = self.contraction_block(apg_2, 128)
        vpg_3, vpg_skip_3 = self.contraction_block(vpg_2, 128)
        ppg_3, ppg_skip_3 = self.contraction_block(ppg_2, 128)

        apg_4, apg_skip_4 = self.contraction_block(apg_3, 256)
        vpg_4, vpg_skip_4 = self.contraction_block(vpg_3, 256)
        ppg_4, ppg_skip_4 = self.contraction_block(ppg_3, 256)
        apg_4 = Dropout(self.config['dropout_1'])(apg_4)
        vpg_4 = Dropout(self.config['dropout_1'])(vpg_4)
        ppg_4 = Dropout(self.config['dropout_1'])(ppg_4)

        bottleneck = self.bottleneck_block(apg_4, vpg_4, ppg_4, 512)
        bottleneck = Dropout(self.config['dropout_1'])(bottleneck)

        exp_4 = self.expansion_block(bottleneck, apg_skip_4, vpg_skip_4, ppg_skip_4, 256)
        exp_3 = self.expansion_block(exp_4, apg_skip_3, vpg_skip_3, ppg_skip_3, 128)
        exp_2 = self.expansion_block(exp_3, apg_skip_2, vpg_skip_2, ppg_skip_2, 64)
        exp_1 = self.expansion_block(exp_2, apg_skip_1, vpg_skip_1, ppg_skip_1, 32)

        abp = self.output_block(exp_1)
        model = Model(inputs=[ppg, vpg, apg], outputs=[abp], name='unet')
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

    def contraction_block(self, input, filters):
        a = 1.67 * filters
        w = [a // 6, a // 3, a // 2]

        x0 = self.basic_block(input, sum(w), 1)
        x1 = self.basic_block(input, w[0], 3)
        x2 = self.basic_block(x1, w[1], 3)
        x3 = self.basic_block(x2, w[2], 3)
        x = Add()([Concatenate()([x1, x2, x3]), x0])
        x_skip = x
        x = MaxPooling1D(pool_size=(2))(x)
        return x, x_skip

    def expansion_block(self, input, apg_skip, vpg_skip, ppg_skip, filters):
        a = 1.67 * filters
        w = [a // 6, a // 3, a // 2]

        input = UpSampling1D(size=2)(input)
        input = Concatenate()([input, ppg_skip, vpg_skip, apg_skip])
        x0 = self.basic_block(input, sum(w), 1)
        x1 = self.basic_block(input, w[0], 3)
        x2 = self.basic_block(x1, w[1], 3)
        x3 = self.basic_block(x2, w[2], 3)
        x = Add()([Concatenate()([x1, x2, x3]), x0])
        return x

    def bottleneck_block(self, apg, vpg, ppg, filters):
        a = 1.67 * filters
        w = [a // 6, a // 3, a // 2]

        input = Concatenate()([ppg, vpg, apg])
        x0 = self.basic_block(input, sum(w), 1)
        x1 = self.basic_block(input, w[0], 3)
        x2 = self.basic_block(x1, w[1], 3)
        x3 = self.basic_block(x2, w[2], 3)
        x = Add()([Concatenate()([x1, x2, x3]), x0])
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
        x = Activation('linear', name='abp')(x)
        return x
