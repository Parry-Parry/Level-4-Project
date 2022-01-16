"""
GRU Seq2Seq with attention

A baseline model to compare to a Transformer for plain text code summary

Based on https://www.tensorflow.org/text/tutorials/nmt_with_attention
"""

"""
TODO:
Decoder
    Call func
Attention
    Entire Obj
Latent Modifiction??
    MLP | Can it be used here
Design argparse format
"""
import Tensorflow as tf


class Attention(object):
    def __init__(self, units) -> None:
        pass


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, encode_units, batch_size, hidden_dim=300) -> None:
        super(Encoder, self).__init__()
        self.units = encode_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        
    def call(self, tokens, state=None):
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's'))
        vectors = self.embedding(tokens)
        shape_checker(vectors, ('batch', 's', 'hidden_dim'))

        output, state = self.gru(vectors, initial_state=state)
        shape_checker(output, ('batch', 's', 'encode_units'))
        shape_checker(state, ('batch', 'encode_units'))

        return output, state


class Decoder(tf.keras.layers.Layer):
    def __init__(self, out_vocab_size, decode_units, batch_size, hidden_dim=300) -> None:
        super(Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.units = decode_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(self.out_vocab_size, hidden_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

        self.attention = Attention(self.units)
        self.convert = tf.keras.layers.Dense(self.units, activation=tf.math.tanh,
                                    use_bias=False)
        self.dense = tf.keras.layers.Dense(self.out_vocab_size)


class seq2seq(tf.keras.Model):
    def __init__(self, encoder, decoder, config, beam_size=None, SOS=None, EOS=None) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.beam_size = beam_size
        self.SOS = SOS
        self.EOS = EOS

    