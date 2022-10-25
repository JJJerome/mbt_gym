from typing import Set

import os

from glob import glob
import importlib


def import_everything() -> None:
    paths = _get_all_py_file_paths()
    failed_paths: Set[str] = set()
    for path in paths:
        try:
            importlib.__import__(_convert_file_path_to_module_path(path), fromlist=[""])
        except ImportError:
            failed_paths.add(path)
    if failed_paths:
        print("Could not import the following files:\n" + "\n".join(failed_paths))
        exit(1)
    else:
        print(f"Successfully imported {len(paths)} .py files.")
        exit(0)


def _get_all_py_file_paths() -> Set[str]:
    all_paths = set(glob(os.path.join("mbt_gym", "**", "*.py"), recursive=True))
    return all_paths


def _convert_file_path_to_module_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    mod_path, basename = parts[:-1], parts[-1]
    basename = basename[:-3]
    return ".".join(mod_path + [basename])


if __name__ == "__main__":
    import_everything()
