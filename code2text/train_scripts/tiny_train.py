import tensorflow as tf
from transformers import KerasMetricCallback, TFAutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, create_optimizer
from datasets import load_dataset, load_metric

import os
import pandas as pd 
import numpy as np 
import pickle

### CUDA SETUP ###

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 3:   
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy =  tf.distribute.get_strategy()

### TOKENIZER ###

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

### CALLBACKS ###

rouge_metric = load_metric("rouge")

def rouge_fn(predictions, labels):
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}

bleu_metric = load_metric("bleu")

def bleu_fn(predictions, labels):
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels)
    return result.items()

### DATASET PREP ###

def tokenize_function(set):
    inputs = tokenizer(set["code"], max_length=512, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
       labels = tokenizer(set["docstring"], max_length=512, padding="max_length", truncation=True)

    inputs["labels"] = labels["input_ids"]

    return inputs

### LOAD DATA ###

train = load_dataset('json', data_files="/users/level4/2393265p/workspace/l4project/data/pyjava/train.jsonl")["train"]
valid = load_dataset('json', data_files="/users/level4/2393265p/workspace/l4project/data/pyjava/valid.jsonl")["train"]

tokenized = train.map(tokenize_function, batched=True)
ds = tokenized.shuffle().train_test_split(test_size=.2)

### ARGS ###

buffer = 512
batch_size = 8
epochs = 4
lr = 4e-4
num_train_steps = len(ds["train"])


### TRAINING ###
"""
with strategy.scope():

    model = TFAutoModelForSeq2SeqLM.from_pretrained("/users/level4/2393265p/workspace/l4project/tinybert", 
    pad_token_id=1, 
    bos_token_id = 0, 
    eos_token_id = 2, 
    decoder_start_token_id = 0)

    optimizer, lr_schedule = create_optimizer(
        init_lr=lr,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=0,
    )

    model.compile(
        optimizer=optimizer
    )
"""

model = TFAutoModelForSeq2SeqLM.from_pretrained("/users/level4/2393265p/workspace/l4project/tinybert", 
    pad_token_id=1, 
    bos_token_id = 0, 
    eos_token_id = 2, 
    decoder_start_token_id = 0)

optimizer, lr_schedule = create_optimizer(
    init_lr=lr,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
    num_warmup_steps=0,
)

model.compile(
    optimizer=optimizer
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

train_set = ds["train"].to_tf_dataset(
                    columns=["input_ids", "attention_mask", "labels"],
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=data_collator)
valid_set = ds["test"].to_tf_dataset(
                    columns=["input_ids", "attention_mask", "labels"],
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=data_collator)
"""
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_set = train_set.with_options(options)   
valid_set = valid_set.with_options(options)    
"""

rouge_callback = KerasMetricCallback(rouge_fn, eval_dataset=valid_set)

bleu_callback = KerasMetricCallback(bleu_fn, eval_dataset=valid_set)

history = model.fit(train_set, epochs=epochs, validation_data=valid_set, callbacks=[rouge_callback, bleu_callback])

pickle.dump(history, open("/users/level4/2393265p/workspace/l4project/tiny/tiny_history.pkl", "wb"))
model.save("/users/level4/2393265p/workspace/l4project/tiny/model")