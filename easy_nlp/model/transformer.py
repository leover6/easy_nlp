import math
import tensorflow as tf
import keras
from keras import layers, initializers


class LayerNorm(layers.Layer):

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        d_size = input_shape[-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=(d_size,),
            initializer=initializers.Ones,
            dtype=tf.float32
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(d_size,),
            initializer=initializers.Zeros,
            dtype=tf.float32
        )

    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)
    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return tf.matmul(p_attn, value), p_attn


class MultiHeadAttention(layers.Layer):

    def __init__(self, h, d_model, dropout=0.1, output_attn=False, **kwargs):
        super().__init__(**kwargs)
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.linears = [layers.Dense(d_model) for _ in range(4)]
        self.dropout = layers.Dropout(dropout)
        self.output_attn = output_attn

    def call(self, query, key, value, use_causal_mask=False):
        nbatches = query.shape[0]
        q_seq_len = query.shape[1]
        k_seq_len = key.shape[1]
        mask = None

        if use_causal_mask:
            # e.g. a 4x4 causal mask:
            # [[0., 1., 1., 1.],
            #  [0., 0., 1., 1.],
            #  [0., 0., 0., 1.],
            #  [0., 0., 0., 0.]]
            mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)

        # 1. linear transformation
        # 2. B x L x d_model -> B x L x n_head x d_k -> B x n_head x L x d_k
        perm = list(range(4))
        perm[1], perm[2] = perm[2], perm[1]
        query, key, value = [
            tf.transpose(tf.reshape(lin(x), (nbatches, -1, self.h, self.d_k)), perm)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = tf.reshape(
                tf.transpose(x, perm),
                (nbatches, -1, self.h * self.d_k)
            )
        x = self.linears[-1](x)

        if self.output_attn:
            return x, self.attn

        return x
    

class BaseAttention(layers.Layer):

    def __init__(self, n_head, d_model, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.layer_norm = LayerNorm()
        self.pre_norm = pre_norm


class GlobalSelfAttention(BaseAttention):

    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        if self.pre_norm:
            return x + self.layer_norm(attn_output)
        return self.layer_norm(x + attn_output)


class CausalSelfAttention(BaseAttention):

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        if self.pre_norm:
            return x + self.layer_norm(attn_output)
        return self.layer_norm(x + attn_output)


class CrossAttention(BaseAttention):

    def call(self, x, context):
        attn_output = self.mha(query=x, key=context, value=context)
        if self.pre_norm:
            return x + self.layer_norm(attn_output)
        return self.layer_norm(x + attn_output)


class FeedForward(layers.Layer):

    def __init__(self, d_model, d_ff, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.seq = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.layer_norm = LayerNorm()
        self.pre_norm = pre_norm

    def call(self, x):
        if self.pre_norm:
            return x + self.layer_norm(self.seq(x))
        return self.layer_norm(x + self.seq(x))


def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.expand_dims(tf.range(length, dtype=tf.float32), -1)      # (seq, 1)
    depths = tf.expand_dims(tf.range(depth, dtype=tf.float32), 0) / depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

    return pos_encoding

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = x.shape[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= math.sqrt(self.d_model)
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class EncoderLayer(layers.Layer):

    def __init__(self, n_head, d_model, d_ff, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = GlobalSelfAttention(n_head, d_model, pre_norm=pre_norm, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, pre_norm=pre_norm, dropout=dropout)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):

    def __init__(self, n_layer, n_head, d_model, d_ff, vocab_size, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(*kwargs)
        self.n_layer = n_layer
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.enc_layers = [
            EncoderLayer(n_head, d_model, d_ff, dropout, pre_norm)
            for _ in range(n_layer)
        ]
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Add dropout.
        x = self.dropout(x)

        for i in range(self.n_layer):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(layers.Layer):

    def __init__(self, n_head, d_model, d_ff, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.cross_attention = CrossAttention(n_head, d_model, dropout, pre_norm)
        self.causal_attention = CausalSelfAttention(n_head, d_model, dropout, pre_norm)
        self.ffn = FeedForward(d_model, d_ff, dropout, pre_norm)

    def call(self, x, context):
        x = self.causal_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, n_layer, n_head, d_model, d_ff, vocab_size, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.dec_layers = [
            DecoderLayer(n_head, d_model, d_ff, dropout, pre_norm)
            for _ in range(n_layer)
        ]
        self.dropout = layers.Dropout(dropout)
    
    def call(self, x, context):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Add dropout.
        x = self.dropout(x)

        for i in range(self.n_layer):
            x = self.dec_layers[i](x, context)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class Transformer(keras.Model):
    def __init__(self, n_layer, n_head, d_model, d_ff, vocab_size, dropout=0.1, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(n_layer, n_head, d_model, d_ff, vocab_size, dropout, pre_norm)
        self.decoder = Decoder(n_layer, n_head, d_model, d_ff, vocab_size, dropout, pre_norm)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        logits = self.output_layer(x)  # (batch_size, target_len, target_vocab_size)
