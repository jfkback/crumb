# Using CRUMB Eval
## Install
```
pip install -e git+https://github.com/jfkback/crumb/crumb_eval
```


## Run Format
CRUMB Eval expects retrieval runs to be in JSONL format, where each line looks like:
```
{
    "query": {
        "id": "[ID FROM CRUMB]",
    },
    "items": [
        {
            "id": "[ID FROM CRUMB]",
            "score": float,
        },
        {
            "id": "[ID FROM CRUMB]",
            "score": float,
        },
        ...
    ]
}
```


## Using CLI
### Getting Help
To see how to use CRUMB Eval you can use the `--help` flag to see the options.
```
crumb-eval --help
```


### Basic Usage
By default, `crumb-eval` assumes your retrieval run is on the passage collection and the evaluation queries.
```
crumb-eval --run-path=/tmp/my_retrieval_run.jsonl --task-name=clinical_trial
```

If either of these is not a valid assumption you can use the flags `--is-validation` or `--is-full-docs`.
```
crumb-eval --run-path=/tmp/my_retrieval_run.jsonl --task-name=clinical_trial --is-validation --is-full-docs
```

To make life simpler, if your run file starts with the name of the task you do not need to pass it via `--task-name`.
```
crumb-eval --run-path=/tmp/clinical_trial_retrieval_run.jsonl
```

Additionally, if you want to save the metrics to a JSON file you can using `--output-path`.
```
crumb-eval --run-path=/tmp/clinical_trial_retrieval_run.jsonl --output-path=/tmp/my_metrics.json
```


### Changing Metrics
In the prior examples only the default metrics are used. If you want to use different metrics you can, by providing them as arguments before the parameters.
```
crumb-eval nDCG@20 R@15 --run-path=/tmp/clinical_trial_retrieval_run.json
```
The metrics should be parsable by `ir_measures.parse_measure` to work. Currently, we manually defined a few metrics as binary or non-binary to properly know when to load the binary measures. If your metric is not in either the non-binary or binary metrics we defined you won't see it in the output. For such uses we suggest using the underlying code instead of the CLI.


### MaxP
MaxP is used for certain datasets that only have document-level labels when the passage collections are evaluated. By default, `--max-p="auto"` which will automatically do the correct MaxP for each task configuration. You can manually override this by doing `--max-p` or `--no-max-p`. Note that for some configurations this will result in an error.



## Using Library
A simple example of using the `evaluate` function is shown below. It follows the same structure as the CLI so you can read about it for more details.
```
from crumb_eval import evaluate

evaluate(
    run_path="/tmp/my_retrieval_run.jsonl",
    task_name="clinical_trial",
    output_path="/tmp/my_metrics.json",
)
```
To see all the options and overview look at the docstring for `evaluate`.