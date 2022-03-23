import os
import re
import pandas as pd
from tensorboard import errors

aug_path = "D:\\PROJECT\\data\\raw_aug"
path_pairs = [["parallel_decl.txt", "parallel_desc.txt"], ["parallel_methods_decl.txt", "parallel_methods_desc.txt"]]

out = []

def specialChar(string):
    string = string.replace("DCNL", "\n")
    string = string.replace("DCSP", " ")
    return string


def removeArgs(x):
    if "DCNL" in x:
        x = x.split("DCNL")[-1]
    x = re.sub("[\(\[].*?[\)\]]", "", x)
    x = x.replace("def", "", 1)
    return x.strip().strip(":")


for pair in path_pairs:
    decls = os.path.join(aug_path, pair[0])
    descs = os.path.join(aug_path, pair[1])

    with open(decls, 'r', errors='ignore') as f:
        decl_data = f.readlines()
    with open(descs, 'r', errors='ignore') as f:
        desc_data = f.readlines()
    
    decl_data = [removeArgs(decl) for decl in decl_data]
    desc_data = [specialChar(desc).strip("'") for desc in desc_data]

    frame = pd.DataFrame(list(zip(decl_data, desc_data)), columns=['code', 'docstring'])
    out.append(frame)

out = pd.concat(out)
out.to_json("aug.jsonl", orient='records', lines=True)

    