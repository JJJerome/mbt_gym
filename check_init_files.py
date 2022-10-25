import os


if __name__ == "__main__":
    """We want to check for missing init files as they may cause tests and mypy to not run"""
    result = [os.path.join(dp, f) for dp, dn, file_names in os.walk(".") for f in file_names]
    result = [r for r in result if r.endswith(".py")]

    dirs = set([os.path.dirname(f) for f in result])
    missing_init_files = []
    for _dir in dirs:
        init_file = os.path.sep.join([_dir, "__init__.py"])
        if not os.path.isfile(init_file):
            missing_init_files.append(init_file)

    if missing_init_files:
        print("Missing init files: ")
        print("\n".join(missing_init_files))
        exit(1)
    else:
        print("No init files seem to be missing")
        exit(0)
