import tensorflow as tf
from transformers import RobertaTokenizer, TFAutoModel

model = TFAutoModel.from_pretrained("distilroberta-base")

class seq2seqHead(tf.keras.layers.Laye):
    def __init__(self, vocab_size, hidden_dim, encode_units) -> None:
        super(seq2seqHead, self).__init__()

class robertaTrain(tf.keras.Model):
    def __init__(self, embedding_dim, units,
               input_text_processor,
               output_text_processor):
        super().__init__()
    
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

