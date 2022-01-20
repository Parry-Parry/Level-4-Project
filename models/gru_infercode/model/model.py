"""
GRU Seq2Seq with attention

A baseline model to compare to a Transformer for plain text code summary

Based on https://www.tensorflow.org/text/tutorials/nmt_with_attention
"""

"""
TODO:
Latent Modifiction??
    MLP | Can it be used here
"""
import typing
from typing import Tuple
from typing import Any

import Tensorflow as tf

"""
## HELPER FUNCTIONS ##
"""

class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    if isinstance(names, str):
      names = (names,)

    shape = tf.shape(tensor)
    rank = tf.rank(tensor)

    if rank != len(names):
      raise ValueError(f'Rank mismatch:\n'
                       f'    found {rank}: {shape.numpy()}\n'
                       f'    expected {len(names)}: {names}\n')

    for i, name in enumerate(names):
      if isinstance(name, int):
        old_dim = name
      else:
        old_dim = self.shapes.get(name, None)
      new_dim = shape[i]

      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key):
    self.key = key
    self.logs = []

  def on_train_batch_end(self, n, logs):
    self.logs.append(logs[self.key])

"""
## CONTAINERS ##
"""

class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any


class DecoderOutput(typing.NamedTuple):
  logits: Any
  attention_weights: Any

"""
## ATTENTION COMPONENTS ##

from https://arxiv.org/pdf/1409.0473.pdf
"""

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):
    shape_checker = ShapeChecker()
    shape_checker(y_true, ('batch', 't'))
    shape_checker(y_pred, ('batch', 't', 'logits'))

    loss = self.loss(y_true, y_pred)
    shape_checker(loss, ('batch', 't'))

    mask = tf.cast(y_true != 0, tf.float32)
    shape_checker(mask, ('batch', 't'))
    loss *= mask

    return tf.reduce_sum(loss)

"""
## MODEL COMPONENTS ##
"""

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

    def call(self,
         inputs: DecoderInput,
         state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'encode_units'))
        shape_checker(inputs.mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        vectors = self.embedding(inputs.new_tokens)
        shape_checker(vectors, ('batch', 't', 'hidden_dim'))

        rnn_output, state = self.gru(vectors, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'decode_units'))
        shape_checker(state, ('batch', 'decode_units'))

        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        shape_checker(context_vector, ('batch', 't', 'decode_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        attention_vector = self.convert(context_and_rnn_output)                       
        shape_checker(attention_vector, ('batch', 't', 'decode_units'))

        logits = self.dense(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weights), state


class infer2seqTrain(tf.keras.Model):
    def __init__(self, embedding_dim, units,
               input_text_processor,
               output_text_processor):
        super().__init__()
    
        encoder = #infercode
        decoder = Decoder(output_text_processor.vocabulary_size(),
                      embedding_dim, units)

        self.encoder = encoder                                                               
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.shape_checker = ShapeChecker()

    def _preprocess(self, input_text, target_text):
        self.shape_checker(input_text, ('batch',))
        self.shape_checker(target_text, ('batch',))

        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        self.shape_checker(input_tokens, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch', 't'))

        input_mask = input_tokens != 0
        self.shape_checker(input_mask, ('batch', 's'))

        target_mask = target_tokens != 0
        self.shape_checker(target_mask, ('batch', 't'))

        return input_tokens, input_mask, target_tokens, target_mask


    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
        self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
        self.shape_checker(dec_state, ('batch', 'decode_units'))

        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state   


    def _train_step(self, inputs):
        input_text, target_text = inputs  

        (input_tokens, input_mask,
        target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            self.shape_checker(enc_output, ('batch', 's', 'encode_units'))
            self.shape_checker(enc_state, ('batch', 'encode_units'))

            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                new_tokens = target_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                             enc_output, dec_state)
                loss = loss + step_loss

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

            variables = self.trainable_variables 
            gradients = tape.gradient(average_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return {'batch_loss': average_loss}


    def train_step(self, inputs):
        return self._train_step(inputs)


    