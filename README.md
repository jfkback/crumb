
# 🍪 CRUMB: A Complex Retrieval Unified Multi-task Benchmark 🍪

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/jfkback/crumb)
[![ArXiv](https://img.shields.io/badge/arXiv-2509.07253-b31b1b.svg)](https://arxiv.org/abs/2509.07253)

**CRUMB** is a diverse and realistic benchmark designed to evaluate the capabilities of information retrieval models on complex, multi-aspect search tasks. It consists of eight meticulously curated retrieval tasks that have multiple components or requirements (i.e. are complex) unlike many common existing evaluation collections and benchmarks.

This repository contains the code, data, and artifacts associated with our paper:

> **Benchmarking Information Retrieval Models on Complex Retrieval Tasks**
> *Julian Killingback and Hamed Zamani*
> [Link to Paper (ArXiv)]([https://arxiv.org/abs/2509.07253](https://arxiv.org/abs/2509.07253))

## Overview

Traditional retrieval benchmarks often focus on simple, single-aspect queries. However, as language models advance, user expectations have shifted towards systems that can understand complex queries with multiple parts, constraints, and requirements. CRUMB was created to bridge this gap by providing a unified evaluation suite for these challenging scenarios.

Our benchmark demonstrates that even state-of-the-art retrieval models struggle with these tasks, highlighting significant room for innovation in next-generation retrieval systems.

### Key Features

*   **Eight Diverse Tasks**: Covers a wide range of domains including legal QA, clinical trials, code, scientific papers, and more.
*   **Complex Queries**: Queries are natural and contain multiple constraints or requirements.
*   **Realistic & Standardized Data**: Documents are provided in a unified Markdown format, with contextualized chunking to preserve document structure.
*   **Passage and Full-Document Versions**: Evaluate models on both chunked passages (for standard retrievers) and full documents (for long-context models).
*   **Validation Sets Included**: Each task includes a development set to enable tuning and few-shot prompting approaches.

## The CRUMB Benchmark Tasks

| Task Name | Query Type | Corpus |
| :--- | :--- | :--- |
| **Paper Retrieval** | Multi-aspect scientific paper criteria | Scientific paper abstracts |
| **Code Retrieval** | Multi-constraint coding problems | Verified code solutions |
| **Theorem Retrieval** | Mathematical problems | Mathematical theorems |
| **Legal QA** | Legal questions with geographic constraints | State legal statutes |
| **Tip-of-the-Tongue** | Vague, multi-detail descriptions of movies/TV | Wikipedia pages |
| **Clinical Trial Retrieval** | Patient medical histories | Clinical trial descriptions |
| **StackExchange QA** | Community questions requiring reasoning | Web pages and Wikipedia |
| **SetOps** | Entity queries with set-based operations | Wikipedia pages |

## Getting Started

The easiest way to get started with the CRUMB dataset is by using the Hugging Face `datasets` library.

### Installation

```bash
pip install datasets
```

### Loading the Data

The dataset is hosted on the Hugging Face Hub at [**jfkback/crumb**](https://huggingface.co/datasets/jfkback/crumb).

You can load any of the eight tasks. Each task is available in two configurations: `passage` (chunked) and `full_document`.

```python
from datasets import load_dataset

# Load the passage-level version of the Clinical Trial task
clinical_trial_passage = load_dataset("jfkback/crumb", "passage_corpus", split="clinical_trial")

# Load the full-document version of the Tip-of-the-Tongue task
tot_full_doc = load_dataset("jfkback/crumb", "full_document_corpus", split="tip_of_the_tongue")

# Loading Queries & Labels
theorem_retrieval_queries_and_qrels = load_dataset("jfkback/crumb", "evaluation_queries", split="theorem_retrieval")
```

## Data Structure
The dataset is organized into four subsets each containing various parts of CRUMB's data. The four subsets are `evaluation_queries`, `validation_queries`, `passage_corpus`, and `full_document_corpus`. Each subset has a split for each task (e.g. clinical_trial). One thing to note is some of the tasks use the same content for the passage and full document subsets. This is the case for tasks where documents were short enough that they did not require chunking.
### *evaluation_queries* and *validation_queries*

These configurations contain the queries and their corresponding ground-truth relevance labels (qrels). Both have the same structure.

* **query_id**: A unique string identifier for the query.
* **query_content**: The string content of the complex query.
* **instruction**: An optional string providing a task-specific instruction (e.g., "Find relevant legal statutes").
* **passage_qrels**: A list of dictionaries containing relevance judgments for the `passage_corpus`.
    * **id**: The `document_id` of a passage from the `passage_corpus`.
    * **label**: A `float32` graded relevance score (higher is more relevant).
* **passage_binary_qrels**: Same as `passage_qrels`, but the `label` is a binary score (1 for relevant, 0 for not). Use these for binary metrics (e.g. Recall) if they are provided otherwise the non-binary ones are fine with a cutoff of relevance > 0.
* **full_document_qrels**: A list of dictionaries containing relevance judgments for the `full_document_corpus`.
    * **id**: The `document_id` of a document from the `full_document_corpus`.
    * **label**: A `float32` graded relevance score.
* **use_max_p**: A boolean flag used to identify whether this query collection should be evaluated with MaxP (where documents are aggrigated by their maximum scoring chunk).
* **metadata**: A stringified JSON object that contains additional metadata about the query. This varies by task.
### *passage_corpus*
This configuration contains the corpus of chunked documents (passages). Note for
* **document_id**: A unique string identifier for the passage.
* **document_content**: The text content of the passage.
* **parent_id**: The `document_id` of the full document from which this passage was extracted, if applicable.
* **metadata**: A stringified JSON object that contains additional metadata about the passage. This varies by task.
### **full_document_corpus**
This configuration contains the corpus of full, un-chunked documents.
* **document_id**: A unique string identifier for the full document.
* **document_content**: The complete text content of the document.
* **parent_id**: Should be None for all documents as they have no parents.
* **metadata**: A stringified JSON object that contains additional metadata about the passage. This varies by task.
## How to Use and Evaluate
Keep these important considerations in mind when evaluating:
* Use the `passage_binary_qrels` if it is available for a task and you are using binary evaluation metrics such as Recall or Precision.
* We highly suggest using MaxP for the passage (i.e. chunked) collection for datasets which do not have per-chunk labels these are:
  * `clinical_trial`
  * `tip_of_the_tongue`
  * `set_operation_entity_retrieval`
* Some of the tasks have the same content for the passage and full document collections, this is because the original documents were too short to chunk so the document is both a passage and a full document. Keep this in mind for evaluation as you do not need to evaluate on both subsets. These datasets are:
  * `code_retrieval`
  * `paper_retrieval`
  * `theorem_retrieval`


## Evaluation

To make benchmarking on CRUMB as simple as possible, we have developed an evaluation library. See the details [here](crumb_eval/README.md).


## Baseline Results & Artifacts

To facilitate further research and replication, we are releasing the artifacts from our baseline experiments.

### Baseline Model Runs

**Status: Coming Soon!**

We will provide the full retrieval runs (e.g., top 2000 documents per query) for all benchmarked models, including BM25, GTE, Promptriever, and Lion. This will allow for direct comparison and reranking experiments without needing to re-run the first-stage retrieval.

### Rewritten Queries

**Status: Coming Soon!**

This release will include the queries generated by our LLM-based rewriting strategies (`Query-to-Answer`, `Query-to-Doc`, `Query-as-Reasoning-Trace`). These can be used to replicate our query augmentation experiments or as a resource for studying query transformation.

### Qualitative Examples

**Status: Coming Soon!**

To provide a better intuition for each task, we will release a sample of queries paired with a relevant document chunk. This can be useful for error analysis and understanding the unique challenges of each task.

## Citing CRUMB

If you use the CRUMB benchmark or any of its associated artifacts in your research, please cite our paper.

```
@misc{killingback2025benchmarkinginformationretrievalmodels,
      title={Benchmarking Information Retrieval Models on Complex Retrieval Tasks},
      author={Julian Killingback and Hamed Zamani},
      year={2025},
      eprint={2509.07253},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.07253},
}
```

Additionally, as CRUMB is composed of several existing datasets, we ask that you also cite the original sources for the specific tasks you use. Please refer to Section 6.4 of our paper for the full list of citations.
