from collections import defaultdict
import os
import pandas as pd


langs = ["python", "java"]
augment_paths = {'python' : "", 'java' : ""}
dataset_path = ""
out_path = ""
filenames = ["train", "test", "valid"]

# LOAD DATA

augment_frame = {lang : pd.read_json(augment_paths[lang]) for lang in langs}

dataset_frame = {file : pd.read_json(os.path.join(dataset_path, file + ".jsonl") ,lines=True) for file in filenames}

# FRAME 2 DICR

def augment_dict(df):
    df.set_index(['identifier'])
    dd = defaultdict(list)
    return df.to_dict('index', into=dd) 

augment_dicts = {lang : augment_dict(augment_frame[lang]) for lang in langs}

# AUGMENT

def augment(row):
    lang = row["language"]
    code_tokens = row.code.split()
    for token in code_tokens:
        augment = augment_dicts[lang][token]
        if augment:
            row.docstring += "\n" + augment["docstring"]

for k in dataset_frame.keys():
    dataset_frame[k] = dataset_frame[k].apply(lambda x : augment(x))

for k, v in dataset_frame.items():
    v.to_json(os.path.join(out_path, k + ".jsonl"), orient='records', lines=True)
