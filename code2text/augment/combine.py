import pandas as pd

# load all augments

augment_paths = {'python' : "", 'java' : ""}
dataset_path = ""
filenames = ["train.jsonl", "test.jsonl", "valid.jsonl"]

# chunksize the datasets

# if indentifier in code tokens
    # append docstring of identifier to docstring of code