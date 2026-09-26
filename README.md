
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
To facilitate further research and replication, we have released the artifacts from our baseline experiments.

### Baseline Model Runs
Below are the retrieval runs (e.g., top 2000 documents per query) for all benchmarked models, including BM25, GTE, Promptriever, and Lion. These runs can be used for direct comparison and reranking experiments without needing to re-run the first-stage retrieval. All the rus are in a Google Drive folder, [here](https://drive.google.com/drive/folders/1x7oJnFBVnQSgGbNw5enWgbdd3VhN58gZ?usp=sharing). You can download the files going through the UI, but I suggest looking at [gdown](https://github.com/wkentaro/gdown) or the Google Drive interface provided by [rclone](https://rclone.org/drive).

The structure of the files in the folder are:
```
├── [TASK NAME]
│   ├── full_document
│   │   ├── no_instruction
│   │   └── none
│   └── passage
│       ├── gemma_3_27b_query_as_reasoning_steps
│       ├── gemma_3_27b_query_as_reasoning_steps_no_instruction
│       ├── gemma_3_27b_query_to_answer_cot
│       ├── gemma_3_27b_query_to_answer_cot_no_instruction
│       ├── gemma_3_27b_query_to_doc_cot
│       ├── gemma_3_27b_query_to_doc_cot_no_instruction
│       ├── no_instruction
│       └── none
```
The `[TASK NAME]` corresponds to the name of the task, `passage` and `full_document` contain the passage and full document results respectively. For tasks with only one set of documents, the results are only shown under the `full_document` directory. Inside the `passage` or `full_document` we have the various query variants used. For most baseline models the main results are in the `none` directory, with the exception of Lion models which have the main results in the `no_instruction` directory. Note that BM25 is not in the `no_instruction` folder, but it did not recieve instructions. There are also other files corresponding to the query rewriting techniques. Some rewriting directories have the suffix `no_instruction` these are the valid (i.e. correspond to the main results) for Lion models only.

The format of retrieval runs is the same as what is expected by the evaluation script. You can find more details [here](crumb_eval/README.md).

### Rewritten Queries
Below are the queries generated by our LLM-based rewriting strategies (`Query-to-Answer`, `Query-to-Doc`, `Query-as-Reasoning-Trace`). These can be used to replicate our query augmentation experiments or as a resource for studying query transformation. All the rewritten queries are in a Google Drive folder, [here](https://drive.google.com/drive/folders/19ZqPYklgS6XKloW9Y1jOVLg1MfabmvaZ?usp=sharing). You can download the files going through the UI, but I suggest looking at [gdown](https://github.com/wkentaro/gdown) or the Google Drive interface provided by [rclone](https://rclone.org/drive).

The folder has the following structure:
```
├── [TASK NAME]
│   ├── queries_gemma_3_27b_query_as_reasoning_steps.jsonl
│   ├── queries_gemma_3_27b_query_to_answer_cot.jsonl
│   ├── queries_gemma_3_27b_query_to_doc_cot.jsonl
│   ├── queries_kimi_query_as_reasoning_steps.jsonl
│   ├── queries_kimi_query_to_answer_cot.jsonl
│   └── queries_kimi_query_to_doc_cot.jsonl
```
Each file is a JSONL where each line contains a rewritten query, in the form:
```
{
   "content": Contains the rewritten query string,
   "id": The ID of the query, this should match the ID of the original query.
   "metadata": A dictionary with any metadata about the generation, generally includes "full_generation" and "prompt".
   "original_item": A dictionary with the original query before being rewritten, has keys "content", "id", and "metadata". 
}
```

The approach descriptions below are copied verbatim from Section 4.2.2 of [Killingback and Zamani, *Benchmarking Information Retrieval Models on Complex Retrieval Tasks*](https://arxiv.org/abs/2509.07253), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Query-to-Answer CoT** Given a query, we prompt the LLM to directly answer the question after an extended "thinking" period. The thinking information is not used.

**Query-to-Document CoT** Given a query, we prompt the LLM to provide a document which provides an answer to the query after an extended "thinking" period. The query instruction provides the expected document type (e.g. a piece of code, web page, etc.). The thinking trace is not used for retrieval only the final document.

**Query as Reasoning Trace** Given a query, the model is asked to understand the query, think about it, and produce a complete answer. Unlike previous approaches, the entire reasoning process is used as part of the query. This is similar to the approach used in BRIGHT (Su et al., 2025).

#### Gemma 3 27B

| Dataset Name | Query-to-Answer | Query-to-Doc | Query-as-Reasoning-Trace |
| :--- | :---: | :---: | :---: |
| Paper Retrieval | [link](https://drive.google.com/open?id=1yPrwP0Ufi6d4HdAqyRg_fkd9AmmmxBrf) | [link](https://drive.google.com/open?id=1-1oxDf_-fSE7L5InMhYLXL_-Y-1Tg2N7) | [link](https://drive.google.com/open?id=1xJWWG3b3LFRZsid_tME7zHpEqAmDR_9M) |
| Code Retrieval | [link](https://drive.google.com/open?id=1IzRYlapyj6yV6p-92_qmZ3naSAL5Tm2M) | [link](https://drive.google.com/open?id=1ZHHHxHFpOdnbuOQIClmmbiUgAIKT8ovC) | [link](https://drive.google.com/open?id=1V6Md1MLOHh8OIHeaNCy_9pjaMMRYFozw) |
| Theorem Retrieval | [link](https://drive.google.com/open?id=1xCyfkjUqIEslEFGE2TR34i1JHSGx3sN8) | [link](https://drive.google.com/open?id=1veOUIfNAkkwhRGPoMagQvPuiNUvhwWLX) | [link](https://drive.google.com/open?id=1e8lWVO4mkQD6nUXWBJJddVFLflMJAh1a) |
| Legal QA | [link](https://drive.google.com/open?id=1PgYvrcWmcvxCQghtoEiBpJblsByJL1QO) | [link](https://drive.google.com/open?id=1pGtxhu5DxtQydrnprofiVFv0tpUF5PXa) | [link](https://drive.google.com/open?id=1c7SewzHw4in-QA0fFSQEW5Dp-n7vP9Pn) |
| Tip-of-the-Tongue | [link](https://drive.google.com/open?id=1UGJkr6tEVh2pHYHwCI1jpZV0y1ngQNTU) | [link](https://drive.google.com/open?id=1YTWJryIp6QDG9BjSUTlCY9Ymh4HKeRP-) | [link](https://drive.google.com/open?id=1ODqzPRwrGcZqdy4_nvnyAHqMf0ONfMkR) |
| Clinical Trial Retrieval | [link](https://drive.google.com/open?id=1afjPNuz9uYJc1iOl6wapyAww0n31K-DP) | [link](https://drive.google.com/open?id=1ofUSXS4OxAZJ4yMdmp2LATQ8jl_4qi2o) | [link](https://drive.google.com/open?id=10n5MNvacoeuiGiOCJDJ-EEAUXrO6YwFV) |
| StackExchange QA | [link](https://drive.google.com/open?id=14HtYDNpSJAw6QPswYWo9jYc8yWUYPjij) | [link](https://drive.google.com/open?id=1X8IGL2pgu559JZqUQSJZB7x3avN1hWKp) | [link](https://drive.google.com/open?id=1TNY_5jOifEU7yomsFKHc5ffByIgP-Vy4) |
| SetOps | [link](https://drive.google.com/open?id=1mVfPH4FwsPyq1_rGXXeyRwn5V67uv3_4) | [link](https://drive.google.com/open?id=1Hzg0v266zCmtLsyz9w1ke-W7sYMkRpCg) | [link](https://drive.google.com/open?id=19HlQQpGlz0PnjAY-ID7ICsdqeypIHXoC) |

#### Kimi

Kimi rewrites for Code Retrieval and Legal QA use random 500-query samples; the other datasets use their full test splits.

| Dataset Name | Query-to-Answer | Query-to-Doc | Query-as-Reasoning-Trace |
| :--- | :---: | :---: | :---: |
| Paper Retrieval | [link](https://drive.google.com/open?id=1PMbBct_ITPPHIBH9snYg5O7f-U6hVUj0) | [link](https://drive.google.com/open?id=16GVZIFMdSGaaRqZtoNPr7l_Spsd3d4GV) | [link](https://drive.google.com/open?id=1SKAIjJpJvWvfdWU_4VaLa9UfKB_n0WrB) |
| Code Retrieval | [link](https://drive.google.com/open?id=16f1cCbLongk2t02D_sVFb6h4SDe27Eez) | [link](https://drive.google.com/open?id=1fP0wxXGgX5ax-kWJvD6LRP-ICIjcI8D0) | [link](https://drive.google.com/open?id=1EF3jBJEVc0Ako4GsKsB_lMHCAnLjP589) |
| Theorem Retrieval | [link](https://drive.google.com/open?id=132wggSb9N8KvlUNC4ae4Es-dIy9Tlzcn) | [link](https://drive.google.com/open?id=1hm4UoEd83QGRMyyL9BjdzLoBHwuYM4W9) | [link](https://drive.google.com/open?id=1YN4FxGjLkEuZh1VAt7Zs7tOLbGsqMMlu) |
| Legal QA | [link](https://drive.google.com/open?id=14M-LgE-TP1iAy3lRmZ6qwQTgKKVX9nGy) | [link](https://drive.google.com/open?id=1Vhj4Q34gx5kyL40RHCeQ2WXW5IETlktc) | [link](https://drive.google.com/open?id=1O5tJ2KioGXLlRk-Liv-Cq0f3beYEwtp5) |
| Tip-of-the-Tongue | [link](https://drive.google.com/open?id=1KRjgSXPIyruwTit3N3qs68j3MDGsJojg) | [link](https://drive.google.com/open?id=1_ZEEddol1buPy9b1v7gqBAKDVdhssTrW) | [link](https://drive.google.com/open?id=1UUGI_I-vd9BVzrrx0Uomr1jhP1jkh53-) |
| Clinical Trial Retrieval | [link](https://drive.google.com/open?id=1CuQpBZzIptHRSaRu1Nt2vFVrI6rfu3ry) | [link](https://drive.google.com/open?id=1pjcdHhlFi8zo3rQc8FagDp31M0KTit_s) | [link](https://drive.google.com/open?id=1UAVRCYBW-BhSjR-l2qItyXlmDa4KXiZ1) |
| StackExchange QA | [link](https://drive.google.com/open?id=1iYNtjeq2M9kXgTgRLrwFl2NBZ5CtDNJQ) | [link](https://drive.google.com/open?id=1RjF2dBhhpniD0pbbRB3AD1QvUB-eN5fX) | [link](https://drive.google.com/open?id=17M1bILljbJY-_S-UNFsSdC5M3o10QVi0) |
| SetOps | [link](https://drive.google.com/open?id=1PAd8shmAiSng-0s410j4Gyxrl0xQApmy) | [link](https://drive.google.com/open?id=1N5278QV-xr4mO-7SAVvutF71JNaMJM08) | [link](https://drive.google.com/open?id=1HwAhZTaWzFi0EXCZP5ZJchg-1y8j7jv0) |

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
