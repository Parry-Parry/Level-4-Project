from collections import defaultdict
import os
import re
import pandas as pd


langs = ["python", "java"]
augment_paths = {'python' : "D:\PROJECT\Augments\python.jsonl", 'java' : "D:\PROJECT\Augments\java.jsonl"}
dataset_path = "D:\PROJECT\data\CodeSearchNet\pyjava"
out_path = "D:\PROJECT\data\CodeSearchNet\\aug_pyjava"
filenames = ["train", "test", "valid"]

# LOAD DATA

augment_frame = {lang : pd.read_json(augment_paths[lang], lines=True) for lang in langs}

dataset_frame = {file : pd.read_json(os.path.join(dataset_path, file + ".jsonl"), lines=True) for file in filenames}

# FRAME 2 DICT

def augment_dict(df):
    df.set_index(['code'])
    dd = defaultdict(list)
    return df.to_dict('index', into=dd) 

print("Dictionary Build...")
augment_dicts = {lang : augment_dict(augment_frame[lang]) for lang in langs}

# AUGMENT
augments = 0
def augment(row):
    lang = row['language']
    code_tokens = row.code.split()
    for token in code_tokens:
        token = re.sub("[\(\[].*?[\)\]]", "", token)
        token = token.strip()
        if "." in token: token = token.split(".")[-1]
        if lang == "java" : 
            token = token.strip(";")
            if "(" in token:
                token = token.split("(")[0]
            if ")" in token:
                token = token.strip(")")
        try:
            augment = augment_dicts[lang][token]  
            if augment:
                print(augment)
                row.docstring += "\n" + augment
                augments += 1
        except KeyError:
            pass
    return row
        
            
print("Augmenting...")
for k in dataset_frame.keys():
    dataset_frame[k] = dataset_frame[k].apply(lambda x : augment(x), axis=1)
print("{} Augments Applied".format(augments))
print("Saving...")
for k, v in dataset_frame.items():
    v.to_json(os.path.join(out_path, k + ".jsonl"), orient='records', lines=True)
