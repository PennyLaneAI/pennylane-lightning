from pathlib import Path
import re
import fnmatch

SRCFILE_EXT = ("c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "cu", "cuh")

LIGHTNING_SOURCE_DIR = Path(__file__).resolve().parent.parent

rgx_gitignore_comment = re.compile("#.*$")

def get_cpp_files_from_path(path, ignore_patterns = None, use_gitignore = True):
    """return set of C++ source files from a path

    Args:
        paths (pathlib.Path or str): a path to process 
        ignore_patterns: patterns to ignore
        use_gitignore: find ignore patterns from .gitignore
    """
    path = Path(path)
    files_rel = set() # file paths relative to path
    for ext in SRCFILE_EXT:
        for file_path in path.rglob(f"*.{ext}"):
            files_rel.add(file_path.relative_to(path))

    if ignore_patterns is None:
        ignore_patterns = []

    if use_gitignore:
        # simple gitignore parser
        gitignore_file = path.joinpath('.gitignore')
        if gitignore_file.exists():
            with gitignore_file.open() as f:
                for line in f.readlines():
                    line = rgx_gitignore_comment.sub('', line)
                    line = line.strip()
                    if line:
                        ignore_patterns.append(line)
    
    files_to_remove = set()
    for ignore_pattern in ignore_patterns:
        for f in files_rel:
            if fnmatch.fnmatch(str(f), ignore_pattern):
                files_to_remove.add(f)

    files_rel -= files_to_remove

    return set(str(path.joinpath(f)) for f in files_rel)
    
def get_cpp_files(paths, ignore_patterns = None, use_gitignore = True):
    """return list of C++ source files from paths.

    Args:
        paths (list): list of all paths to process
        ignore_patterns: patterns to ignore
        use_gitignore: find ignore patterns from .gitignore
    """

    files = set()
    for path in paths:
        files |= get_cpp_files_from_path(path, ignore_patterns, use_gitignore)
    return list(files)
