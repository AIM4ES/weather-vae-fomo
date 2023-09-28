from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import Model
import math

from tensorflow.keras.layers import (
    Dense,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Reshape,
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Flatten,
)

class ConvDecoder(Model):
    def __init__(self, filter_list, flat_shape, agg_size, kernel_size) -> None:
        super(ConvDecoder, self).__init__()

        self.layer_list = []
        self.decoder_input_layer = Dense(
            math.prod(flat_shape), activation="relu"
        )
        self.reshape = Reshape(flat_shape)
        filter_list.append(1)

        for filter in filter_list[1:]:
            self.layer_list.append(UpSampling2D(size=(agg_size, 1)))
            self.layer_list.append(
                Conv2DTranspose(
                    filters=filter,
                    kernel_size=(1, kernel_size),
                    strides=1,
                    padding="same",
                )
            )
            self.layer_list.append(BatchNormalization())
            self.layer_list.append(LeakyReLU())

        self.final_layer = Conv2D(filters=1, kernel_size=3, padding="same")

    def call(self, inputs):
        x = self.decoder_input_layer(inputs)
        x = self.reshape(x)
        for layer in self.layer_list:
            x = layer(x)
        x = self.final_layer(x)

        return x

    def model(self, input_shape: tuple):
        """
        A model call just to have the input shape visible in summary

        Args:
            input_shape (tuple): Tuple having an input shape

        Returns:
            tf.keras.Model: The model
        """
        x = tf.keras.layers.Input(shape=input_shape)

        return Model(inputs=[x], outputs=self.call(x))

class ConvEncoder(Model):
    def __init__(self, filter_list, latent_dim, agg_size, kernel_size) -> None:
        super(ConvEncoder, self).__init__()

        self.layer_list = []

        for filter in filter_list:
            self.layer_list.append(
                Conv2D(
                    filters=filter,
                    kernel_size=(3, kernel_size),
                    strides=1,
                    padding="same",
                )
            )
            self.layer_list.append(
                MaxPooling2D(pool_size=(agg_size, 1), padding="same")
            )
            self.layer_list.append(BatchNormalization())
            self.layer_list.append(LeakyReLU(alpha=0.3))

        self.layer_list.append(Flatten())
        self.fc_mu = Dense(latent_dim)
        self.fc_logvar = Dense(latent_dim)

    def call(self, inputs):

        x = inputs

        for layer in self.layer_list:
            x = layer(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def model(self, input_shape: tuple):
        """
        A model call just to have the input shape visible in summary

        Args:
            input_shape (tuple): Tuple having an input shape

        Returns:
            tf.keras.Model: The model
        """
        x = tf.keras.layers.Input(shape=input_shape)

        return Model(inputs=[x], outputs=self.call(x))


class SharedDecoderConvVAE(Model):
    def __init__(
        self,
        latent_dim: int,
        inp_shape: Tuple,
        filter_list: List,
        agg_size: int,
        kernel_size: int,
        **kwargs
    ) -> None:

        super(SharedDecoderConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.inp_shape = inp_shape
        self.filter_list = filter_list
        self.agg_size = agg_size
        self.kernel_size = kernel_size

        self.encoder_block = ConvEncoder(
            self.filter_list, self.latent_dim, self.agg_size, self.kernel_size
        )
        self.encoder_block.model(self.inp_shape).summary()

        flat_shape = (
            inp_shape[0] // (agg_size) ** (len(self.filter_list)),
            inp_shape[1] // 3,
            filter_list[-1],
        )

        self.filter_list.reverse()

        self.decoder_block = ConvDecoder(
            self.filter_list, flat_shape, self.agg_size, self.kernel_size
        )

        self.decoder_block.model((self.latent_dim // 2,)).summary()

    def reparametrize(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        std = tf.math.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        return eps * std + mu

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mu, log_var = self.encoder_block(inputs)
        z = self.reparametrize(mu, log_var)

        sm_z_dim = self.latent_dim//4

        z_rad = z[:, :sm_z_dim]
        z_precip = z[:, sm_z_dim:2*sm_z_dim]
        z_temp = z[:, 2*sm_z_dim:3*sm_z_dim]
        z_cov = z[:, 3*sm_z_dim:]

        r_rad = self.decoder_block(tf.concat([z_rad, z_cov], axis=1))
        r_precip = self.decoder_block(tf.concat([z_precip, z_cov], axis=1))
        r_temp = self.decoder_block(tf.concat([z_temp, z_cov], axis=1))
        r = tf.concat([r_rad, r_precip, r_temp], axis=2)
        return z, r, mu, log_var

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "inp_shape": self.inp_shape,
            "filter_list": self.filter_list,
            "agg_size": self.agg_size,
            "kernel_size": self.kernel_size,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)