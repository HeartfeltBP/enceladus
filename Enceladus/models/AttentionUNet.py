import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout, Add, Lambda, Multiply
from keras.initializers.initializers_v2 import GlorotUniform, HeUniform


class AttentionUNet():
    def __init__(self, config):
        self.config = config
        ini, act = self.get_model_components(self.config)
        self.ini = ini
        self.act = act

    def init(self):
        ppg = Input(shape=(256, 1), name='ppg')
        vpg = Input(shape=(256, 1), name='vpg')
        apg = Input(shape=(256, 1), name='apg')

        filt = 32

        # encoder 1 : 1 x 256 -> 32 x 128
        apg_1, apg_skip_1 = self.contraction_block(apg, filters=filt)  # 32
        vpg_1, vpg_skip_1 = self.contraction_block(vpg, filters=filt)
        ppg_1, ppg_skip_1 = self.contraction_block(ppg, filters=filt)

        # encoder 2 : 32 x 128 -> 64 x 64
        apg_2, apg_skip_2 = self.contraction_block(apg_1, filters=filt * 2)  # 64
        vpg_2, vpg_skip_2 = self.contraction_block(vpg_1, filters=filt * 2)
        ppg_2, ppg_skip_2 = self.contraction_block(ppg_1, filters=filt * 2)

        # encoder 3 : 64 x 64 -> 128 x 32
        apg_3, apg_skip_3 = self.contraction_block(apg_2, filters=filt * 4)  # 128
        vpg_3, vpg_skip_3 = self.contraction_block(vpg_2, filters=filt * 4)
        ppg_3, ppg_skip_3 = self.contraction_block(ppg_2, filters=filt * 4)

        # encoder 4 : 128 x 32 -> 256 x 16
        apg_4, apg_skip_4 = self.contraction_block(apg_3, filters=filt * 8)  # 256
        vpg_4, vpg_skip_4 = self.contraction_block(vpg_3, filters=filt * 8)
        ppg_4, ppg_skip_4 = self.contraction_block(ppg_3, filters=filt * 8)

        # dropout
        apg_skip_4 = Dropout(self.config['dropout'])(apg_skip_4)
        vpg_skip_4 = Dropout(self.config['dropout'])(vpg_skip_4)
        ppg_skip_4 = Dropout(self.config['dropout'])(ppg_skip_4)
        apg_4 = Dropout(self.config['dropout'])(apg_4)
        vpg_4 = Dropout(self.config['dropout'])(vpg_4)
        ppg_4 = Dropout(self.config['dropout'])(ppg_4)

        # bottlneck : 768 x 16 -> 512 x 16
        bottleneck = self.bottleneck_block(apg_4, vpg_4, ppg_4, filters=filt * 16)  # 512

        # dropout
        bottleneck = Dropout(self.config['dropout'])(bottleneck)

        # decoder 4 : 512 x 16 -> 256 x 32
        dec_4 = self.expansion_block(bottleneck, apg_skip_4, vpg_skip_4, ppg_skip_4, filters=filt * 8)  # 256

        # decoder 3 : 256 x 32 -> 128 x 64
        dec_3 = self.expansion_block(dec_4, apg_skip_3, vpg_skip_3, ppg_skip_3, filters=filt * 4)  # 128

        # decoder 2 : 128 x 64 -> 64 x 128
        dec_2 = self.expansion_block(dec_3, apg_skip_2, vpg_skip_2, ppg_skip_2, filters=filt * 2)  # 64

        # decoder 1 : 64 x 128 -> 32 x 256
        dec_1 = self.expansion_block(dec_2, apg_skip_1, vpg_skip_1, ppg_skip_1, filters=filt)  # 32

        # output : 32 x 256 -> 1 x 256 
        abp = self.output_block(dec_1)

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
        if config['initializer'] != 'None':
            ini = initializers[config['initializer']]
        else:
            ini = None
        if config['activation'] != 'None':
            act = activations[config['activation']]
        else:
            act = None
        return ini, act

    def basic_block(self, input, filters, size):
        x = Conv1D(
            filters=filters,
            kernel_size=(size),
            kernel_initializer=self.ini,
            padding='same',
        )(input)
        x = BatchNormalization()(x)
        x = Activation(self.act)(x)
        return x

    def contraction_block(self, input, filters):
        x = self.basic_block(input, filters, 3)
        x = self.basic_block(x, filters, 3)
        skip = x
        x = MaxPooling1D(pool_size=(2))(x)
        return x, skip

    def bottleneck_block(self, apg, vpg, ppg, filters):
        x = Concatenate()([ppg, vpg, apg])
        x = self.basic_block(x, filters, 3)
        x = self.basic_block(x, filters, 3)
        return x

    def repeat(self, input, n):
        x = Lambda(lambda x, iter: K.repeat_elements(x, iter, axis=2), arguments=dict(iter=n))(input)
        return x

    def attention_block(self, input, context, filters):
        context_1 = Conv1D(
            filters=filters,
            kernel_size=(1),
            padding='same',
        )(context)
        input_1 = Conv1D(
            filters=filters,
            kernel_size=(3),
            strides=K.int_shape(input)[1] // K.int_shape(context)[1],
            padding='same',
        )(input)

        input_context = Add()([context_1, input_1])
        input_context = Activation(self.act)(input_context)

        psi = Conv1D(
            filters=1,
            kernel_size=(1),
            padding='same',
        )(input_context)
        psi = Activation('sigmoid')(psi)

        alpha = UpSampling1D(
            size=K.int_shape(input)[1] // K.int_shape(psi)[1],
        )(psi)
        alpha = self.repeat(alpha, K.int_shape(input)[2])

        coef = Multiply()([alpha, input])
        x = Conv1D(
            filters=K.int_shape(input)[2],
            kernel_size=1,
            padding='same',
        )(coef)
        x = BatchNormalization()(x)
        return x

    def expansion_block(self, input, apg_skip, vpg_skip, ppg_skip, filters):
        skip = Concatenate()([ppg_skip, vpg_skip, apg_skip])
        att = self.attention_block(skip, input, filters * 2)

        x = UpSampling1D(size=2)(input)
        x = self.basic_block(x, filters, 2)

        x = Concatenate()([att, x])
        x = self.basic_block(x, filters, 3)
        x = self.basic_block(x, filters, 3)
        return x

    def output_block(self, input):
        x = Conv1D(
            filters=2,
            kernel_size=(3),
            kernel_initializer=self.ini,
            padding='same',
        )(input)
        x = Activation(self.act)(x)
        x = Conv1D(
            filters=1,
            kernel_size=(3),
            kernel_initializer=self.ini,
            padding='same'
        )(x)
        x = Activation('linear', name='abp')(x)
        return x
