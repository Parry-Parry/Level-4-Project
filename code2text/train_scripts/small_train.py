from transformers import TFAutoModelForSeq2SeqLM, RobertaTokenizer, TrainingArguments, Trainer

import tensorflow as tf

import os
import pandas as pd 
import numpy as np 
import pickle

ROOT_DIR = "X"

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 1:   
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy =  tf.distribute.get_strategy()

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert")

model = TFAutoModelForSeq2SeqLM.from_pretrained("/users/level4/2393265p/workspace/l4project/code/smallbert/smallbert")

def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, return_tensors='tf')

train = pd.read_json("/users/level4/2393265p/workspace/l4project/data/pyjava/train.jsonl", lines=True)
valid = pd.read_json("/users/level4/2393265p/workspace/l4project/data/pyjava/valid.jsonl", lines=True)

batch_size = 32
buffer = 2048

X_train = tokenize_function(train["code"].values)
y_train = tokenize_function(train["docstring"].values)
X_val = tokenize_function(valid["code"].values)
y_val = tokenize_function(valid["docstring"].values)

with strategy.scope():
    train_set = (
        tf.data.Dataset.from_tensor_slices(
            (   
                X_train,
                y_train
            )
        )
    ).shuffle(buffer).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    valid_set = (
        tf.data.Dataset.from_tensor_slices(
            (
                X_val,
                y_val
            )  
        )
    ).shuffle(buffer).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=4e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy()
    )

    history = model.fit(train_set, epochs=4, validation_data=valid_set)

pickle.dump(history, open("/users/level4/2393265p/workspace/l4project/small/small_history.pkl", "wb"))
model.save("/users/level4/2393265p/workspace/l4project/small/model")

