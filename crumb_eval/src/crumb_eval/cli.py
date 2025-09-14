from pathlib import Path
from typing import Literal

import cyclopts

from crumb_eval.eval.eval import evaluate

app = cyclopts.App(help="A CLI tool to run crumb evaluations.")


@app.default
def evaluate_cli(
    *metrics: str,
    run_path: str | Path,
    task_name: str | None = None,
    output_path: str | Path | None = None,
    max_p: bool | Literal["auto"] = "auto",
    is_validation: bool = False,
    is_full_docs: bool = False,
) -> None:
    """Evaluates a retrieval run on a specified CRUMB task.

    Args:
        metrics (str): A set of positional arguments that represent metrics to
            compute. These metrics must be strings parsable by
            `ir_measures.parse_measure`. If none are provided, a default set of
            metrics will be used.
        run_path (str | Path): The path to the JSONL run file. If this file is
            named with a task prefix (e.g., `clinical_trial_run1.jsonl`), the
            task name can be inferred and does not need to be provided. Otherwise,
            the task name must be provided with the `task_name` argument.
        task_name (str | None, optional): The name of the CRUMB task. This
            should match the Huggingface split e.g. `clinical_trial`. This is
            used to know which qrels to load. This is not needed if the prefix
            of the `run_path` can be used to infer the task name. Defaults to
            None.
        output_path (str | Path | None, optional): The path to the output JSON
            file. If not provided the metrics are only printed not stored.
            Defaults to None.
        max_p (bool | Literal["auto"], optional): Whether to use max_p
            evaluation. If True, MaxP will try to be used when valid. If False,
            MaxP will not be used. If "auto" is used the behavior is determined
            automatically, based on what was originally used in the CRUMB
            paper. We highly recommend using "auto". Defaults to "auto".
        is_validation (bool, optional): Whether to compute metrics for the
            validation set. Defaults to False.
        is_full_docs (bool, optional): Whether the run is on full documents.
            Defaults to False.
    """

    if len(metrics) == 0:
        metrics = None

    evaluate(
        run_path=run_path,
        task_name=task_name,
        output_path=output_path,
        metrics=metrics,
        max_p=max_p,
        is_validation=is_validation,
        is_full_docs=is_full_docs,
    )


def main():
    app()


if __name__ == "__main__":
    main()
