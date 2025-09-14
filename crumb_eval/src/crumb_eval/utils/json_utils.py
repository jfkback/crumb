from contextlib import contextmanager
from pathlib import Path

import jsonlines

from crumb_eval.utils.path_utils import resolve_path


@contextmanager
def JsonlReader(filepath: str | Path, **kwargs):
    filepath = resolve_path(filepath)
    with jsonlines.open(filepath, **kwargs) as reader:
        yield reader


@contextmanager
def JsonlWriter(
    filepath: str | Path, check_exists: bool = False, mode: str = "w", **kwargs
):
    filepath = resolve_path(filepath)

    path_object = Path(filepath)
    if check_exists and path_object.exists():
        raise FileExistsError(f"File {filepath} already exists.")

    if not path_object.parent.exists():
        path_object.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(filepath, mode=mode, **kwargs) as writer:
        yield writer
