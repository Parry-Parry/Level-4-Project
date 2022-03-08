from tree_sitter import Language

languages = [
    'grammars/tree-sitter-python',
    'grammars/tree-sitter-java'
]

Language.build_library(
    # Store the library in the directory
    'lang/py-tree-sitter-languages.so',
    # Include one or more languages
    languages
)
