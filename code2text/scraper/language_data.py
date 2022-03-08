from parsers.java_parser import JavaParser
from parsers.python_parser import PythonParser



LANGUAGE_METADATA = {
    'python': {
        'platform': 'pypi',
        'ext': 'py',
        'language_parser': PythonParser
    },
    'java': {
        'platform': 'maven',
        'ext': 'java',
        'language_parser': JavaParser
    }
}
