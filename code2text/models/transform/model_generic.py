from typing import Tuple
import tensorflow as tf
import tensorflow_addons as tfa

from transformers import TFRobertaModel, RobertaTokenizer, RobertaConfig

from code2text.models.baseline.model import DecoderInput, DecoderOutput

MODELS = {'roberta':()}


class seq2seq():
    def __init__(self, type, model_name) -> None:
        tokenizer, model, config_type = MODELS[type]

        embedding_size = 3072
        config = config_type.from_pretrained(model_name, output_hidden_states=True)
        self.encoder = model.from_pretrained(model_name, config=config)
        self.encoder.roberta.trainable = False
        self.tokenizer = tokenizer.from_pretrained("microsoft/codebert")

        self.embedding = tf.keras.layers.Embedding(self.tokenizer.vocab_size, embedding_size)

        decoder_cell = tf.keras.layers.LSTMCell(embedding_size)
        sampler = tfa.seq2seq.TrainingSampler()
        output_layer = tf.keras.layers.Dense(self.tokenizer.vocab_size)
        self.decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer)
    

    