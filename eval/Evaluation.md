# Evaluation of VolCano

## Data Prepare
For evaluation, we also utilize the yaml file to manage the evaluation datasets.

For CLEVR and CLEVR-ref, we provide the constructed evaluation meta data in [here](https://huggingface.co/datasets/luoruipu1/VoCoT/tree/main/eval). For other dataets, please refer to the original website for the data.

With the prepared datasets, please set the correct paths in those [config files](../config/datasets/eval/).

## Run Evaluation

We provide the evaluation scripts in [test_all_benchmark.sh](./commands/test_all_benchmark.sh), you can modify the model path and run the entire script or use a part of it.

## Metric Computation

Similar to the evaluation, all the metrics can be computed offline with [run_metric.sh](./commands/run_metric.sh). For GQA and AMBER, the output is converted into appropriate format. You need to further compute the metric, please refer to [LLaVA_for_GQA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#gqa) and [AMBER](https://github.com/junyangwang0410/AMBER) for further instruction.
