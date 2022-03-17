import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric

import os
import pandas as pd 
import numpy as np 
import pickle

### MODEL INIT ###

tmp = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained("sshleifer/tiny-distilroberta-base", "sshleifer/tiny-distilroberta-base")
tmp.save_pretrained("tiny_model")

### TOKENIZER ###

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

### CALLBACKS ###
"""
metric = load_metric("bleu", "rouge", "meteor")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
"""
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

tokenized_train = train.map(tokenize_function, batched=True)
tokenized_valid = valid.map(tokenize_function, batched=True)

train_set = tokenized_train.shuffle()
valid_set = tokenized_valid.shuffle()

### ARGS ###

batch_size = 64
epochs = 4
lr = 4e-4

### TRAINING ###

model = AutoModelForSeq2SeqLM.from_pretrained("tiny_model", 
    pad_token_id=1, 
    bos_token_id = 0, 
    eos_token_id = 2, 
    decoder_start_token_id = 0)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="/users/level4/2393265p/workspace/l4project/models/tiny/model_trained",
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)

trainer.train()