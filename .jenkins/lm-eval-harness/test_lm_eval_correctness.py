"""
LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml

* export LM_EVAL_TEST_DATA_FILE=configs/Meta-Llama-3-70B-Instruct.yaml
* export LM_EVAL_TP_SIZE=4 
* pytest -s test_lm_eval_correctness.py
"""
import atexit
import os
from pathlib import Path

import lm_eval
import numpy
import yaml

RTOL = 0.05
TEST_DATA_FILE = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    ".jenkins/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct.yaml")

TP_SIZE = os.environ.get("LM_EVAL_TP_SIZE", 1)


def fail_on_exit():
    os._exit(1)


def launch_lm_eval(eval_config):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    dtype = eval_config.get('dtype', 'bfloat16')
    max_num_seqs = eval_config.get('max_num_seqs', 128)
    model_args = f"pretrained={eval_config['model_name']}," \
                 f"tensor_parallel_size={TP_SIZE}," \
                 f"add_bos_token=true," \
                 f"dtype={dtype}," \
                 f"max_model_len=4096," \
                 f"max_num_seqs={max_num_seqs}," \
                 f"trust_remote_code={trust_remote_code}"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size="auto")

    return results


def test_lm_eval_correctness():
    eval_config = yaml.safe_load(
        Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

    # Launch eval requests.
    results = launch_lm_eval(eval_config)

    # Confirm scores match ground truth.
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f'{task["name"]} | {metric["name"]}: '
                  f'ground_truth={ground_truth} | measured={measured_value}')
            try:
                assert numpy.isclose(ground_truth, measured_value, rtol=RTOL)
            except AssertionError as exc:
                # nasty workaround for HPU PT bridge bug (SW-204785)
                atexit.register(fail_on_exit)
                raise exc
