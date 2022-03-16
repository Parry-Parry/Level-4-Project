import os
import sys
import pandas as pd 
import numpy as np 
import pickle

import tensorflow as tf

sys.path.append('/users/level4/2393265p/workspace/l4project/code/Level-4-Project')

from code2text.models.baseline.model import seq2seqTrain, MaskedLoss
from code2text.helper.model import BatchLogs
from code2text.helper.preprocess import tf_lower_and_split_punct

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 1:   
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy =  tf.distribute.get_strategy()
"""
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"""
tf.config.run_functions_eagerly(False)


train = pd.read_json("/users/level4/2393265p/workspace/l4project/data/pyjava/train.jsonl", lines=True)
valid = pd.read_json("/users/level4/2393265p/workspace/l4project/data/pyjava/valid.jsonl", lines=True)

batch_size = 64 
buffer = 2048

train_set = (
    tf.data.Dataset.from_tensor_slices(
        (   
            tf.cast(train["code"].values, tf.string),
            tf.cast(train["docstring"].values, tf.string)
        )
    )
).shuffle(buffer).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
valid_set = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(valid["code"].values, tf.string),
            tf.cast(valid["docstring"].values, tf.string)
        )  
    )
).shuffle(buffer).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

input_processor = input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    vocabulary="/users/level4/2393265p/workspace/l4project/data/outvocab.txt")

output_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    vocabulary="/users/level4/2393265p/workspace/l4project/data/outvocab.txt")

train_model = seq2seqTrain(112, 64, input_text_processor=input_processor,
    output_text_processor=output_processor, strategy=strategy)

train_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=8e-4),
    loss=MaskedLoss(),
    metrics=['acc']
)

batch_loss = BatchLogs('batch_loss')
chkpt = tf.keras.callbacks.ModelCheckpoint(
    "/users/level4/2393265p/workspace/l4project/baseline/chkpt", monitor='acc', save_best_only=False, save_freq=2000
)

history = train_model.fit(train_set, epochs=4, validation_data=valid_set, callbacks=[batch_loss])

"""
with strategy.scope():

    input_processor = input_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        vocabulary="/users/level4/2393265p/workspace/l4project/data/outvocab.txt")

    output_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        vocabulary="/users/level4/2393265p/workspace/l4project/data/outvocab.txt")

    train_model = seq2seqTrain(112, 64, input_text_processor=input_processor,
        output_text_processor=output_processor, strategy=strategy)

    train_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=4e-4),
        loss=MaskedLoss(),
        metrics=['acc']
    )

    batch_loss = BatchLogs('batch_loss')
    chkpt = tf.keras.callbacks.ModelCheckpoint(
        "/users/level4/2393265p/workspace/l4project/baseline/chkpt", monitor='acc', save_best_only=False, save_freq=2000
    )

history = train_model.fit(train_set, epochs=3, validation_data=valid_set, callbacks=[batch_loss, chkpt])
"""
pickle.dump(history, open("/users/level4/2393265p/workspace/l4project/baseline/baseline_hisory.pkl", "wb"))
pickle.dump(batch_loss['logs'], open("/users/level4/2393265p/workspace/l4project/baseline/baseline_loss.pkl", "wb"))
train_model.save("/users/level4/2393265p/workspace/l4project/baseline/model")