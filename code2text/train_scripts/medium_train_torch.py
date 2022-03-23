import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from datasets import load_dataset

import os
import pandas as pd 
import numpy as np 
import pickle

### TOKENIZER ###

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

### EVAL METRICS ###

bleu = datasets.load_metric('bleu')
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=predictions, references=references, rouge_types=["rouge2"])["rouge2"].mid
    bleu_output = bleu.compute(predictions=[pred.split() for pred in predictions], references=[[ref.split()] for ref in references])
    meteor_output = meteor.compute(predictions=predictions, references=references)


    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        "bleu_score" : bleu_output["bleu"],
        "meteor_score" : meteor_output["meteor"]
    }

def eval_compute(results):
    predictions=results["pred_string"] 
    references=results["docstring"]

    rouge_output = rouge.compute(predictions=predictions, references=references, rouge_types=["rouge2"])["rouge2"].mid
    bleu_output = bleu.compute(predictions=[pred.split() for pred in predictions], references=[[ref.split()] for ref in references])
    meteor_output = meteor.compute(predictions=predictions, references=references)

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        "bleu_score" : bleu_output["bleu"],
        "meteor_score" : meteor_output["meteor"]
    }


### DATASET PREP ###

def tokenize_function(set):
    inputs = tokenizer(set["code"], max_length=512, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
       labels = tokenizer(set["docstring"], max_length=512, padding="max_length", truncation=True)

    inputs["labels"] = labels["input_ids"].copy()
    inputs["decoder_input_ids"] = labels.input_ids
    inputs["decoder_attention_mask"] = labels.attention_mask

    inputs["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in inputs["labels"]]

    return inputs

### LOAD DATA ###

train = load_dataset('json', data_files="/users/level4/2393265p/workspace/l4project/data/py_clean/train.jsonl")["train"]
valid = load_dataset('json', data_files="/users/level4/2393265p/workspace/l4project/data/py_clean/valid.jsonl")["train"]
test = load_dataset('json', data_files="/users/level4/2393265p/workspace/l4project/data/py_clean/test.jsonl")["train"]

tokenized_train = train.map(tokenize_function, batched=True, remove_columns=train.column_names)
tokenized_valid = valid.map(tokenize_function, batched=True, remove_columns=valid.column_names)

train_set = tokenized_train.shuffle()
valid_set = tokenized_valid.shuffle()

train_set.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
valid_set.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

### ARGS ###

batch_size = 4
epochs = 6
lr = 4e-4

### CONFIG ###

model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained("nyu-mll/roberta-med-small-1M-1", "nyu-mll/roberta-med-small-1M-1")

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 4

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

### TRAINING ###

training_args = Seq2SeqTrainingArguments(
    output_dir="/users/level4/2393265p/workspace/l4project/models/medium/model_trained",
    predict_with_generate=True,
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    weight_decay=0.01,
    save_total_limit=1,
    save_steps=10000,
    num_train_epochs=epochs,
    logging_steps=2000,
    overwrite_output_dir=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

### EVALUATION ###

def generate_string(batch):
    inputs = tokenizer(batch["code"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_string"] = output_str
    return batch

trainer.save_model("/users/level4/2393265p/workspace/l4project/models/medium/model_out")

results = test.map(generate_string, batched=True, batch_size=batch_size)

results = eval_compute(results)

with open("/users/level4/2393265p/workspace/l4project/models/medium/results.pkl", "wb") as f:
    pickle.dump(results, f)


