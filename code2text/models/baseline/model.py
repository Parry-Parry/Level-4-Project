"""
GRU Seq2Seq with attention

A baseline model to compare to a Transformer for plain text code summary

Based on https://www.tensorflow.org/text/tutorials/nmt_with_attention
"""

import typing
from typing import Tuple
from typing import Any

import tensorflow as tf
import numpy as np

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

        with tf.device('CPU:0'):
            self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        w1_query = self.W1(query)
        w2_key = self.W2(value)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )

        return context_vector, attention_weights


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):

    loss = self.loss(y_true, y_pred)

    mask = tf.cast(y_true != 0, tf.float32)
    loss *= mask

    return tf.reduce_sum(loss)

"""
## MODEL COMPONENTS ##
"""

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_dim, encode_units) -> None:
        super(Encoder, self).__init__()
        self.out_vocab_size = vocab_size
        self.units = encode_units
        with tf.device('CPU:0'):
            self.embedding = tf.keras.layers.Embedding(self.out_vocab_size, hidden_dim)

        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        
    def call(self, tokens, state=None):
        vectors = self.embedding(tokens)
        output, state = self.gru(vectors, initial_state=state)
        return output, state


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_dim, decode_units) -> None:
        super(Decoder, self).__init__()
        self.out_vocab_size = vocab_size
        self.units = decode_units
        with tf.device('cpu:0'):
            self.embedding = tf.keras.layers.Embedding(self.out_vocab_size, hidden_dim)
            self.attention = Attention(self.units)
            self.dense = tf.keras.layers.Dense(self.out_vocab_size)

        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        self.convert = tf.keras.layers.Dense(self.units, activation=tf.math.tanh,
                                    use_bias=False)
        
    def call(self,
         inputs: DecoderInput,
         state=None) -> Tuple[DecoderOutput, tf.Tensor]:

        vectors = self.embedding(inputs.new_tokens)

        rnn_output, state = self.gru(vectors, initial_state=state)

        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)

        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        attention_vector = self.convert(context_and_rnn_output)                       

        logits = self.dense(attention_vector)

        return DecoderOutput(logits, attention_weights), state


class seq2seqTrain(tf.keras.Model):
    def __init__(self, embedding_dim, units,
               input_text_processor,
               output_text_processor):
        super().__init__()
    
        encoder = Encoder(input_text_processor.vocabulary_size(),
                      embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                      embedding_dim, units)

        self.encoder = encoder                                                               
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

    def _preprocess(self, input_text, target_text):

        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        input_mask = input_tokens != 0
        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask


    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

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

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def train_step(self, inputs):
        return self._train_step(inputs)


class seq2seq(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor,
               output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)

        result_text = tf.strings.reduce_join(result_text_tokens,
                                       axis=1, separator=' ')

        result_text = tf.strings.strip(result_text)
        return result_text
    
    def sample(self, logits, temperature):
        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else: 
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits/temperature,
                                        num_samples=1)

        return new_tokens

    def translate_unrolled(self,
                       input_text, *,
                       max_length=50,
                       return_attention=True,
                       temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                             enc_output=enc_output,
                             mask=(input_tokens!=0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, input_text):
        return self.translate_unrolled(input_text)



    