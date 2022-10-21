import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Dropout


class UNet():
    def __init__(self) -> None:
        pass

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

    def contraction_block(self, input, filters, dropout=False):
        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        if dropout:
            x = Dropout(rate=0.5)(x)
            skip = x
        else:
            skip = x

        x = MaxPooling1D(pool_size=(2))(x)
        return x, skip

    def bottleneck_block(self, input, filters):
        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Dropout(rate=0.5)(x)
        x = UpSampling1D(size=2)(x)
        return x

    def expansion_block(self, input, skip, filters):
        x = Conv1D(filters=filters, kernel_size=(2), padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Concatenate()([x, skip])

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = UpSampling1D(size=2)(x)
        return x

    def output_block(self, input, skip, filters):
        x = Conv1D(filters=filters, kernel_size=(2), padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Concatenate()([x, skip])

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv1D(filters=filters, kernel_size=(3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv1D(filters=2, kernel_size=(3), padding='same')(x)
        output = Conv1D(filters=1, kernel_size=(3), padding='same')(x)
        return output
