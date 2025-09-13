from pathlib import Path
from typing import List, Literal

import cyclopts
import datasets
from datasets import load_dataset

from crumb_eval.eval.data_representations import Item, QueryAssociatedItems
from crumb_eval.utils.json_utils import JsonlReader, JsonlWriter

TASKS = [
    "clinical_trial",
    "code_retrieval",
    "legal_qa",
    "paper_retrieval",
    "set_operation_entity_retrieval",
    "stack_exchange",
    "theorem_retrieval",
    "tip_of_the_tongue",
]

FULL_DOC_TASKS = [
    "clinical_trial",
    "legal_qa",
    "set_operation_entity_retrieval",
    "stack_exchange",
    "tip_of_the_tongue",
]

USE_MAX_P = [
    "clinical_trial",
    "set_operation_entity_retrieval",
    "tip_of_the_tongue",
]

HAS_BINARY_QRELS = [
    "paper_retrieval",
]


def load_relevant_query_label_dataset(
    task_name: str, is_validation: bool, is_full_docs: bool, is_binary: bool
) -> datasets.Dataset | datasets.DatasetDict:
    if is_full_docs and task_name not in FULL_DOC_TASKS:
        print(
            f"Warning: Task {task_name} only has one collection, ignoring full_docs flag."
        )
        is_full_docs = False

    if is_binary and task_name not in HAS_BINARY_QRELS:
        is_binary = False

    if is_binary and is_full_docs:
        is_binary = False

    split = "validation" if is_validation else "evaluation"
    full_split_name = f"{split}_queries"

    return load_dataset("jfkback/crumb", full_split_name, split=task_name)


def get_qrel_key(is_full_docs: bool, is_binary: bool) -> str:
    qrel_key = ["full_document"] if is_full_docs else ["passage"]

    if is_binary:
        qrel_key.append("binary")

    qrel_key.append("qrels")
    return "_".join(qrel_key)


def load_run(run_path: str | Path) -> List[QueryAssociatedItems]:
    run_data = []
    with JsonlReader(run_path) as reader:
        for obj in reader:
            run_data.append(QueryAssociatedItems.from_dict(obj))
    return run_data


def load_qrels(
    task_name: str, is_validation: bool, is_full_docs: bool, is_binary: bool
) -> List[QueryAssociatedItems]:
    dataset = load_relevant_query_label_dataset(
        task_name=task_name,
        is_validation=is_validation,
        is_full_docs=is_full_docs,
        is_binary=is_binary,
    )

    qrel_key = get_qrel_key(is_full_docs=is_full_docs, is_binary=is_binary)

    def process_example(example) -> QueryAssociatedItems:
        query = Item(
            id=example["query_id"], content=example.get("query_content")
        )
        items, item_labels = [], []
        for info in example[qrel_key]:
            item = Item(id=info["id"])
            items.append(item)
            item_labels.append(info["label"])

        return QueryAssociatedItems(
            query=query, items=items, item_scores=item_labels
        )

    return [process_example(example) for example in dataset]


def determine_task(run_path: str | Path, task_name: str | None = None):
    if task_name is not None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task name: {task_name}")
        return task_name

    run_path = Path(run_path)
    for task in TASKS:
        if run_path.name.startswith(task):
            return task

    raise ValueError(
        f"Could not determine task from run path: {run_path}. Please specify the task name."
    )


def modify_run_for_max_p(
    run_data: List[QueryAssociatedItems],
) -> List[QueryAssociatedItems]:
    pass


def evaluate(
    run_path: str | Path,
    task_name: str | None = None,
    output_path: str | Path | None = None,
    max_p: bool | Literal["auto"] = "auto",
    metrics: List[str] | None = None,
    is_validation: bool = False,
    is_full_docs: bool = False,
) -> None:
    task_name = determine_task(run_path=run_path, task_name=task_name)

    # Handle MaxP logic. If max is true for a task that supports it and the
    # evaluation is on passages we load the full document qrels and need to
    # modify the run to only include the top passage per document.

    if max_p == "auto":
        if is_full_docs:
            # Never use max_p when evaluating on full documents
            max_p = False
        elif task_name not in USE_MAX_P:
            # Conditions: is passage, but task shouldn't use max_p
            max_p = False
        elif task_name in USE_MAX_P:
            # Conditions: is passage and task should use max_p
            max_p = True

    if max_p is True:
        if is_full_docs:
            raise ValueError(
                "max_p cannot be used when evaluating on full documents"
            )

        if task_name not in USE_MAX_P:
            raise ValueError(
                f"max_p is only supported for tasks: {USE_MAX_P}, but got {task_name}"
            )

        print(f"Using MaxP evaluation for task {task_name}")
        qrel_is_full_docs = True
    else:
        qrel_is_full_docs = is_full_docs

    binary_qrels = load_qrels(
        task_name=task_name,
        is_validation=is_validation,
        is_full_docs=qrel_is_full_docs,
        is_binary=True,
    )

    qrels = load_qrels(
        task_name=task_name,
        is_validation=is_validation,
        is_full_docs=qrel_is_full_docs,
        is_binary=False,
    )

    run_data = load_run(run_path=run_path)

    if max_p is True:
        run_data = modify_run_for_max_p(run_data=run_data)

    if metrics is None:
        metrics = ["map", "mrr", "ndcg_cut_10", "recall_10", "recall_100"]

    eval_metrics = compute_metrics(
        run=run_data,
        qrels=qrels,
        binary_qrels=binary_qrels,
        metrics=metrics,
        use_max_p=max_p,
    )

    # Print metrics
    # and/or write to output path


def evaluate_cli(
    *metrics: str,
    run_path: str | Path,
    task_name: str | None = None,
    output_path: str | Path | None = None,
    is_validation: bool = False,
    is_full_docs: bool = False,
) -> None:
    print(metrics)
    print(run_path)
    print(task_name)
    print(output_path)
    print(is_validation)
    print(is_full_docs)


if __name__ == "__main__":
    cyclopts.run(evaluate_cli)
