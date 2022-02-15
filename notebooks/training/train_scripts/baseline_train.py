import os
import io
import sys
import pandas as pd 
import numpy as np 

import tensorflow as tf
import tensorflow_text as text

sys.path.append('CHANGE ME')
from code2text.models.baseline.model import seq2seqTrain, MaskedLoss
from code2text.helper.model import BatchLogs
from code2text.helper.preprocess import tf_lower_and_split_punct

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True) # NEEDS TO BE FOR ALL GPUS
tf.config.run_functions_eagerly(False)

# CHANGE PATHS
train = pd.read_json("D:\PROJECT\data\CodeSearchNet\Combine_clean\\train.json")
valid = pd.read_json("D:\PROJECT\data\CodeSearchNet\Combine_clean\\valid.json")
test = pd.read_json("D:\PROJECT\data\CodeSearchNet\Combine_clean\\test.json")

batch_size = 256
buffer = 2048

train_set = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(train["code"].values, tf.string),
            tf.cast(train["docstring"].values, tf.string)
        )
    )
).shuffle(buffer).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
test_set = (
    tf.data.Dataset.from_tensor_slices(
        (
        tf.cast(test["code"].values, tf.string),
        tf.cast(test["docstring"].values, tf.string)
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

# CHANGE PATHS
input_processor = input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    vocabulary="D:\PROJECT\Level-4-Project\data\outvocab.txt")

output_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    vocabulary="D:\PROJECT\Level-4-Project\data\outvocab.txt")

train_model = seq2seqTrain(112, 64, input_text_processor=input_processor,
    output_text_processor=output_processor)

train_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0004),
    loss=MaskedLoss(),
    metrics=['acc', text.metrics.rouge_l]
)

# CHANGE PATHS
batch_loss = BatchLogs('batch_loss')
chkpt = tf.keras.callbacks.ModelCheckpoint(
    "D:\PROJECT\Level-4-Project\notebooks\training\chkpt\baseline", monitor='loss', save_best_only=True, save_freq=1000
)

history = train_model.fit(train_set, epochs=6, validation_data=valid_set, callbacks=[batch_loss, chkpt])

# WRITE INFO TO FILE