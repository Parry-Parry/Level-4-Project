"""
Retrieve all code in repo
Strip inline comments
create mapping from function string x -> docs regarding x

TODO:
    Tensorflow
    Scipy
    Standard python libs
    Java std
    Java apache
"""
import sys
import os
import shutil as sh


import pandas as pd 

def flatten_strip(path, file_type=".py") -> None:
    """
    TODO:
        Walk path looking for file type
        capture files move to flat directory
    """
    file_dir = os.path.join(path, "code")
    for _, _, files in os.walk(path):
        for file in files:
            if file.lower().endwith(file_type):
                tmp = os.path.join(path, file)
                new = os.path.join(file_dir, file)
                sh.move(tmp, new)

def extract(file, file_type=".py") -> pd.DataFrame:
    pass
    """
    read file
    split by function / class definition
    get comments
    map to function name
    return data frame
    """

def scrape(dir, file_type=".py") -> pd.Dataframe:
    file_dir = os.path.join(dir, "code")
    frame = pd.DataFrame()

    for _, _, files in os.walk(file_dir):
        for file in files:
            tmp = extract(file, file_type)
            frame.append(tmp)
    return frame


def main(path, out, file_type):
    flatten_strip(path, file_type)
    frame = scrape(path, file_type)
    frame.to_json(out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Too few args, provide in path, out path & file type")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])