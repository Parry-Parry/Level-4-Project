from transformers import RobertaTokenizer, TFRobertaForCausalLM
import tensorflow as tf

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_dim, decode_units) -> None:
        super(Decoder, self).__init__()
        self.out_vocab_size = vocab_size
        self.units = decode_units
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

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = TFRobertaForCausalLM.from_pretrained("roberta-base")



