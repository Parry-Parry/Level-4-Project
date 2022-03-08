
import sys
import os


import pandas as pd 
from tree_sitter import Language, Parser
from language_data import LANGUAGE_METADATA
from parsers.language_parser import tokenize_docstring

class dataprocessor():
    PARSER = Parser()

    def __init__(self, language, parser) -> None:
        self.PARSER.set_language(language)
        self.lang_parser = parser


    def get_function_definitions(self, filepath: str):
        try:
            with open(filepath) as source_code:
                print("reading")
                blob = source_code.read()
            tree = self.PARSER.parse(blob.encode())
            return self.lang_parser.get_definition(tree, blob)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError, RecursionError):
            print("error")
            return None
    
    def extract_function_data(self, function):
            return {
                'identifier': function['identifier'],
                'parameters': function.get('parameters', ''),
                'argument_list': function.get('argument_list', ''),
                'return_statement': function.get('return_statement', ''),
                'docstring': function['docstring'].strip(),
                'docstring_summary': function['docstring_summary'].strip(),
                'docstring_tokens': tokenize_docstring(function['docstring_summary']),
                'function': function['function'].strip(),
            }   
        
def extract(path, file_type):
    language = file_type
    lang = Language("scraper/lang/py-tree-sitter-languages.so", language)
    processor = dataprocessor(lang, LANGUAGE_METADATA[file_type]['language_parser'])

    functions = []
    
    for root, _, files in os.walk(path):
        for file in files:
            source = os.path.join(root, file)
            definition = processor.get_function_definitions(source)
            if definition:
                functions += definition
    
    return pd.DataFrame([processor.extract_function_data(func) for func in functions])

def main(path, out, file_type):
    frame = extract(path, file_type)
    print("Saving...")
    if not None:
        frame.to_json(out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Too few args, provide in path, out path, type")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])