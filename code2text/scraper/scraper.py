import sys
import os
import shutil as sh
from pathlib import Path


def flatten_strip(path, sink_dir, type = ".py") -> None:
    file_dir = sink_dir
    i = 0
    print("file dir:", file_dir)
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(type):
                i += 1
                source = os.path.join(root, file)
                sink = os.path.join(file_dir,str(i)+file)
                if not os.path.isfile(sink):
                    sh.copyfile(source, sink)


def main(path, type):
    
    sink = os.path.join(path, "code")
    os.chdir(path)
    if not os.path.exists(sink):
        os.makedirs(sink)
    os.chdir(sink)
    flatten_strip(path, sink, type)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Too few args, provide in path")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])