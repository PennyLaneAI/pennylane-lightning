from pathlib import Path

SRCFILE_EXT = ("c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "cu", "cuh")

LIGHTNING_SOURCE_DIR = Path(__file__).resolve().parent.parent

def get_cpp_files(paths, ignore_patterns = None):
    files = set()

    for path in paths:
        for ext in SRCFILE_EXT:
            for file_path in LIGHTNING_SOURCE_DIR.joinpath(path).rglob(f"*.{ext}"):
                files.add(str(file_path))
    
    if ignore_patterns is not None:
        files_to_remove = set()
        for f in files:
            for ignore_pattern in ignore_patterns:
                if ignore_pattern in f:
                    files_to_remove.add(f)
        files -= files_to_remove
    return list(files)


