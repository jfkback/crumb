import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import datasets
import ir_measures
from datasets import load_dataset
from ir_measures import RR, P, R, nDCG

from crumb_eval.eval.data_representations import Item, QueryAssociatedItems
from crumb_eval.utils.json_utils import JsonlReader

BINARY_METRICS = [type(m) for m in [P, R, RR]]


METRICS = [type(m) for m in [nDCG]]


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
    task_name: str, is_validation: bool, is_binary: bool, is_full_docs: bool
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

    dataset = load_dataset("jfkback/crumb", full_split_name, split=task_name)
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
    def process_query_associated_items(
        query_associated_items: QueryAssociatedItems,
    ) -> QueryAssociatedItems:
        assert query_associated_items.item_scores is not None

        seen_doc_ids = set()
        new_items = []
        new_item_scores = []

        sorted_items_with_scores = sorted(
            zip(
                query_associated_items.items,
                query_associated_items.item_scores,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        for item, score in sorted_items_with_scores:
            assert item.id is not None

            doc_id = item.id.split(":")[0]
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                new_items.append(item)
                new_item_scores.append(score)

        return QueryAssociatedItems(
            query=query_associated_items.query,
            items=new_items,
            item_scores=new_item_scores,
        )

    return [process_query_associated_items(qai) for qai in run_data]


def query_associated_items_to_dict(
    query_associated_items: List[QueryAssociatedItems],
    score_to_int: bool = False,
) -> Dict:
    output = defaultdict(dict)
    for qai in query_associated_items:
        assert qai.item_scores is not None
        for item, score in zip(qai.items, qai.item_scores):
            output[qai.query.id][item.id] = (
                int(score) if score_to_int else score
            )
    return output


def compute_metrics(
    run: List[QueryAssociatedItems],
    qrels: List[QueryAssociatedItems],
    binary_qrels: List[QueryAssociatedItems],
    metrics: List[str],
) -> Dict:
    # Convert to dict for easy lookup
    qrel_dict = query_associated_items_to_dict(qrels, score_to_int=True)
    binary_qrel_dict = query_associated_items_to_dict(
        binary_qrels, score_to_int=True
    )
    run_dict = query_associated_items_to_dict(run, score_to_int=False)

    metric_objects = [ir_measures.parse_measure(metric) for metric in metrics]
    binary_metrics_objects = [
        m for m in metric_objects if type(m) in BINARY_METRICS
    ]
    non_binary_metric_objects = [
        m for m in metric_objects if type(m) in METRICS
    ]

    binary_computed_metrics = ir_measures.calc_aggregate(
        binary_metrics_objects, binary_qrel_dict, run_dict
    )
    non_binary_computed_metrics = ir_measures.calc_aggregate(
        non_binary_metric_objects, qrel_dict, run_dict
    )

    computed_metrics = {
        **binary_computed_metrics,
        **non_binary_computed_metrics,
    }

    return computed_metrics


def print_metrics(computed_metrics: Dict) -> None:
    for metric, value in computed_metrics.items():
        print(f"{metric}: {value:.4f}")


def evaluate(
    run_path: str | Path,
    task_name: str | None = None,
    output_path: str | Path | None = None,
    max_p: bool | Literal["auto"] = "auto",
    metrics: Iterable[str] | None = None,
    is_validation: bool = False,
    is_full_docs: bool = False,
) -> None:
    """Evaluates the retrieval run for the task and collection (passage or full
    document) specified by the run path and task name.

    Args:
        run_path (str | Path): The path to the JSONL run file. If this file is
            named with a task prefix (e.g., `clinical_trial_run1.jsonl`), the
            task name can be inferred and does not need to be provided. Otherwise,
            the task name must be provided with the `task_name` argument.
        task_name (str | None, optional): The name of the task. Used to know
            which qrels to load. This is not needed if the prefix of the `run_path`
            can be used to infer the task name. Defaults to None.
        output_path (str | Path | None, optional): The path to the output JSON
            file. If not provided the metrics are only printed not stored.
            Defaults to None.
        max_p (bool | Literal["auto"], optional): Whether to use max_p
            evaluation. If True, MaxP will try to be used when valid. If False,
            MaxP will not be used. If "auto" is used the behavior is determined
            automatically, based on what was originally used in the CRUMB
            paper. We highly recommend using "auto". Defaults to "auto".
        metrics (Iterable[str] | None, optional): The metrics to compute. These
            should be parsable by `ir_measures.parse_measure`. If not provided,
            a default set of metrics will be used. They are:
              - "nDCG@10"
              - "nDCG@5"
              - "P@10"
              - "P@5"
              - "R@10"
              - "MRR"
              - "R@1000"
              - "R@100"
              - "MRR@10"
            Defaults to None.
        is_validation (bool, optional): Whether to compute metrics for the
            validation set. Defaults to False.
        is_full_docs (bool, optional): Whether the run is on full documents.
            Defaults to False.

    Raises:
        ValueError: If `max_p` is used when evaluating on full documents.
        ValueError: If `max_p` is not supported for the given task.
    """

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
                "`max_p` cannot be used when evaluating on full documents"
            )

        if task_name not in USE_MAX_P:
            raise ValueError(
                f"`max_p` is only supported for tasks: {USE_MAX_P}, but got {task_name}"
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
        metrics = [
            "nDCG@10",
            "nDCG@5",
            "P@10",
            "P@5",
            "R@10",
            "MRR",
            "R@1000",
            "R@100",
            "MRR@10",
        ]

    computed_metrics = compute_metrics(
        run=run_data,
        qrels=qrels,
        binary_qrels=binary_qrels,
        metrics=metrics,
    )

    # Print metrics
    print(f"Metrics for task `{task_name}`:")
    print_metrics(computed_metrics)

    # and/or write to output path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(computed_metrics, f, indent=2)
        print(f"Wrote metrics to {output_path}")
