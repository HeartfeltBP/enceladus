import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout, LSTM, ConvLSTM1D
from keras.initializers.initializers_v2 import GlorotUniform, HeUniform
from keras.regularizers import L1, L2, L1L2


class MultiModalUNet():
    def __init__(self, config):
        self.config = config
        ini, act, reg = self.get_model_components(self.config)
        self.ini = ini
        self.act = act
        self.reg = reg

    def init(self):
        ppg = Input(shape=(256, 1), name='ppg')
        vpg = Input(shape=(256, 1), name='vpg')

        vpg_1, vpg_skip_1 = self.contraction_block(vpg, filters=64, pooling=True)
        vpg_2, vpg_skip_2 = self.contraction_block(vpg_1, filters=128, pooling=True)
        vpg_3, vpg_skip_3 = self.contraction_block(vpg_2, filters=256, pooling=True)
        vpg_4 = self.contraction_block(vpg_3, filters=512, pooling=False)
        vpg_4 = Dropout(self.config['dropout_1'])(vpg_4)
        vpg_skip_4 = vpg_4
        vpg_4 = MaxPooling1D(pool_size=(2))(vpg_4)

        ppg_1, ppg_skip_1 = self.contraction_block(ppg, filters=64, pooling=True)
        ppg_2, ppg_skip_2 = self.contraction_block(ppg_1, filters=128, pooling=True)
        ppg_3, ppg_skip_3 = self.contraction_block(ppg_2, filters=256, pooling=True)
        ppg_4 = self.contraction_block(ppg_3, filters=512, pooling=False)
        ppg_4 = Dropout(self.config['dropout_1'])(ppg_4)
        ppg_skip_4 = ppg_4
        ppg_4 = MaxPooling1D(pool_size=(2))(ppg_4)

        exp_1 = self.contraction_block(ppg_4, filters=1024, pooling=False)
        exp_1 = Dropout(self.config['dropout_2'])(exp_1)
        exp_1 = UpSampling1D(size=2)(exp_1)

        exp_2 = self.expansion_block(exp_1, ppg_skip_4, vpg_skip_4, filters=512, sampling=True)
        exp_3 = self.expansion_block(exp_2, ppg_skip_3, vpg_skip_3, filters=256, sampling=True)
        exp_4 = self.expansion_block(exp_3, ppg_skip_2, vpg_skip_2, filters=128, sampling=True)
        exp_5 = self.expansion_block(exp_4, ppg_skip_1, vpg_skip_1, filters=64, sampling=False)
        abp = self.output_block(exp_5)
        model = Model(inputs=[ppg, vpg], outputs=[abp], name='unet')
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
        res_skip = input
        x = self.basic_block(input, filters, 3)
        x = self.basic_block(x, filters, 3)
        x = Concatenate()([x, res_skip])
        if pooling:
            skip = x
            x = MaxPooling1D(pool_size=(2))(x)
            return x, skip
        else:
            return x

    def expansion_block(self, input, ppg_skip, abp_skip, filters, sampling):
        x = self.basic_block(input, filters, 2)
        x = Concatenate()([x, ppg_skip, abp_skip])
        res_skip = x
        x = self.basic_block(x, filters, 3)
        x = self.basic_block(x, filters, 3)
        x = Concatenate()([x, res_skip])
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