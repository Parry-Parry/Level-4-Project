
import sys
import os

import ast

import pandas as pd 

class dataprocessor():

    def __init__(self) -> None:
        pass

    def get_function_definitions(self, filepath: str):
        file = open(filepath, "r")
        try:
            module = ast.parse(file.read())
            class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
            definitions = []
            for class_def in class_definitions:
                function_definitions = [node for node in class_def.body if isinstance(node, ast.FunctionDef)]
                for f in function_definitions:
                    if ast.get_docstring(f) is not None:
                        definitions.append({'code': f.name, 'docstring': ast.get_docstring(f)})    
            file.close()
            return definitions 
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError, RecursionError, SyntaxError):
            file.close()
            return None
        
def extract(path):

    processor = dataprocessor()

    functions = []
    print("scanning...")
    for root, _, files in os.walk(path):
        for file in files:
            source = os.path.join(root, file)
            definition = processor.get_function_definitions(source)
            if definition:
                functions += definition
    return pd.DataFrame(functions)

def main(path, out):
    frame = extract(path)
    print("Saving...")
    if not None:
        frame.to_json(out)


if __name__ == "__main__":
    """
    args:
        1 : in path to scan
        2 : write path, should be json file
        3 : language type e.g python or java
    """
    if len(sys.argv) < 3:
        print("Too few args, provide in path, out path")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])