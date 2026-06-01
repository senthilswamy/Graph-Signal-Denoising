import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class GraphConvolution(Layer):

    def __init__(self, units, A_hat):
        super().__init__()

        self.units = units
        self.A_hat = A_hat

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):

        x = tf.matmul(inputs, self.w)

        output = tf.einsum(
            'ij,bjk->bik',
            self.A_hat,
            x
        )

        return tf.nn.relu(output)


def DIGAN_Block(
        inputs,
        dropout,
        attention_heads,
        dense_units):

    x = LayerNormalization()(inputs)

    x = Dropout(dropout)(x)

    attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=16
    )(x, x)

    x = Add()([x, attention])

    ffn = Dense(
        dense_units,
        activation='relu'
    )(x)

    ffn = Dense(inputs.shape[-1])(ffn)

    return Add()([x, ffn])


def build_model(
        params,
        A_hat,
        num_nodes):

    lr = float(params[0])
    gcn_units = int(params[2])
    latent_dim = int(params[3])
    dropout = float(params[4])

    inputs = Input(
        shape=(num_nodes, 1)
    )

    digan = DIGAN_Block(
        inputs,
        dropout,
        4,
        128
    )

    g1 = GraphConvolution(
        gcn_units,
        A_hat
    )(inputs)

    g2 = GraphConvolution(
        gcn_units,
        A_hat
    )(g1)

    g3 = GraphConvolution(
        gcn_units,
        A_hat
    )(g2)

    fusion = Concatenate()([digan, g3])

    fusion = Dense(
        128,
        activation='relu'
    )(fusion)

    x = Flatten()(fusion)

    encoder = Dense(
        256,
        activation='relu'
    )(x)

    z_mean = Dense(latent_dim)(encoder)

    z_log_var = Dense(latent_dim)(encoder)

    def sampling(args):

        z_mean, z_log_var = args

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return (
            z_mean +
            tf.exp(0.5*z_log_var)
            * epsilon
        )

    z = Lambda(sampling)(
        [z_mean, z_log_var]
    )

    decoder = Dense(
        256,
        activation='relu'
    )(z)

    decoder = Dense(
        num_nodes,
        activation='linear'
    )(decoder)

    outputs = Reshape(
        (num_nodes,1)
    )(decoder)

    model = Model(
        inputs,
        outputs
    )

    reconstruction_loss = tf.reduce_mean(
        tf.square(inputs - outputs)
    )

    kl_loss = -0.5 * tf.reduce_mean(
        1 +
        z_log_var -
        tf.square(z_mean) -
        tf.exp(z_log_var)
    )

    model.add_loss(
        reconstruction_loss +
        0.001 * kl_loss
    )

    model.compile(
        optimizer=Adam(lr)
    )

    return model