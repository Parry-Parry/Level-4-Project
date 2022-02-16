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
import function_parser
from function_parser.language_data import LANGUAGE_METADATA
from function_parser.process import DataProcessor
from tree_sitter import Language


def flatten_strip(path, file_type=".py") -> None:
    if file_type.lower().equals(".py") or file_type.lower().equals(".java"):
        file_dir = os.path.join(path, "code")
        for _, _, files in os.walk(path):
            for file in files:
                if file.lower().endwith(file_type):
                    tmp = os.path.join(path, file)
                    new = os.path.join(file_dir, file)
                    sh.move(tmp, new)
    else:
        print("File type not currently supported")


def extract(dir, file_type=".py") -> pd.DataFrame:

    if file_type.lower().equals(".py"):
        language = "python"
    if file_type.lower().equals(".java"):
        language = "java"
    else:
        print("File type not currently supported")
        return None
    
    DataProcessor.PARSER.set_language(
    Language(os.path.join(function_parser.__path__[0], "tree-sitter-languages.so"), language))
    processor = DataProcessor(
        language=language, language_parser=LANGUAGE_METADATA[language]["language_parser"]
    )

    path = os.path.join(dir, "code")
    out = processor.process_dee(path, ext=LANGUAGE_METADATA[language]["ext"])
    return out.drop(["nwo", "sha", "path", "language", "docstring_tokens", "function_tokens", "url"])
   

def main(path, out, file_type):
    flatten_strip(path, file_type)
    frame = extract(path, file_type)
    frame.to_json(out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Too few args, provide in path, out path & file type")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])