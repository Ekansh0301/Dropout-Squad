from pathlib import Path

def print_tree(directory: Path, prefix: str = ""):
    entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension)

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    print(root.name)
    print_tree(root)