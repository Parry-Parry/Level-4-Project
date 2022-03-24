from collections import defaultdict
import os
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import load_metric

MAX_AUG = 2

bleu = load_metric("sacrebleu")

langs = ["python", "java"]
#augment_paths = {'python' : "D:\\PROJECT\\Level-4-Project\\code2text\\augment\\aug.jsonl", 'java' : "D:\PROJECT\Augments\java.jsonl"}
augment_path = "D:\\PROJECT\\Augments\\aug.jsonl"
dataset_path = "D:\PROJECT\data\CodeSearchNet\python"
out_path = "D:\PROJECT\data\CodeSearchNet\\aug_python_bleu"
filenames = ["train", "test", "valid"]

# LOAD DATA

#augment_frame = {lang : pd.read_json(augment_paths[lang], lines=True) for lang in langs}

dataset_frame = {file : pd.read_json(os.path.join(dataset_path, file + ".jsonl"), lines=True) for file in filenames}

# FRAME 2 DICT

def augment_dict(df):
    df.set_index(['code'], inplace=True)
    dd = defaultdict(str)
    return df.to_dict('index', into=dd) 

print("Dictionary Build...")
#augment_dicts = {lang : augment_dict(augment_frame[lang]) for lang in langs}
augment_frame = pd.read_json(augment_path, lines=True)
augment_frame = augment_frame.drop_duplicates(subset='code', keep="first")
augmenter = augment_dict(augment_frame)
print("number of potential augmentations: ", len(augment_frame))

# AUGMENT
augments = 0
per_block = []
def augment(row):
    global augments
    code_tokens = row.code_tokens
    code_tokens = [token for token in code_tokens if token.isalnum()]
    for token in code_tokens:
        try:
            token_augment = augmenter[token]
            if token_augment:
                row.docstring += "\n" + token_augment['docstring']
                augments += 1
        except KeyError:
            pass
    return row
        
def relevance_augment(row):
    global augments
    potential = []
    candidate_scores = {}
    code_tokens = row.code_tokens
    code_tokens = [token for token in code_tokens if token.isalnum()]
    for token in code_tokens:
        try:
            token_augment = augmenter[token]
            if token_augment:
                potential.append(token_augment['docstring'])
        except KeyError:
            pass
    if potential:
        for string in potential:
            score = bleu.compute(predictions=[string], references=[[row["docstring"]]])["score"]
            candidate_scores[score] = string

        scores = list(candidate_scores.keys())
        scores.sort(reverse=True)

        if len(scores) > MAX_AUG:
            scores = scores[:MAX_AUG]
        augment_strings = [candidate_scores[score] for score in scores]

        for aug in augment_strings:
            row.docstring += "\n" + aug
            augments += 1
        per_block.append(len(augment_strings))
    else:
        per_block.append(0)
    return row

print("Augmenting...")
for k in tqdm(dataset_frame.keys()):
    print("\nCurrent Augment: {}".format(k), flush=True)
    dataset_frame[k] = dataset_frame[k].apply(lambda x : relevance_augment(x), axis=1)
print("{} Augments Applied".format(augments))
print("Saving...")
for k, v in dataset_frame.items():
    v.to_json(os.path.join(out_path, k + ".jsonl"), orient='records', lines=True)
