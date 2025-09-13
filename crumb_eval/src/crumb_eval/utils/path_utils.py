import os
from pathlib import Path


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def resolve_relative_path(
    relative_path: str, script_file_path: str | Path
) -> str:
    return str(os.path.join(os.path.dirname(script_file_path), relative_path))
