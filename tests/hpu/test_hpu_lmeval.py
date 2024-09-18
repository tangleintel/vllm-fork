import numpy as np

import pytest
import statistics
import lm_eval
from dataclasses import replace

task_manager = lm_eval.tasks.TaskManager(include_path="")


class LMTask:
    def __init__(self, lm_instance, task_cfg):
        self.lm_instance = lm_instance
        self.task_cfg = task_cfg
        assert "task_name" in self.task_cfg, "Task config must contain a task_name!"
        self.task_name = self.task_cfg["task_name"]
        self.task_dict = lm_eval.tasks.get_task_dict(self.task_name, task_manager)
        if "task_config_overrides" in self.task_cfg:
            self.task_dict[self.task_name]._config = replace(
                self.task_dict[self.task_name]._config, **self.task_cfg["task_config_overrides"]
            )

    def run_evaluate(self):
        results = lm_eval.evaluator.evaluate(
            lm=self.lm_instance.LM,
            task_dict=self.task_dict,
            **self.task_cfg["eval_kwargs"]
        )
        return results


class LMInstance:
    def __init__(self, lm_instance_cfg):
        #        self.LM = lm_eval.models.vllm_causallms.VLLM(
        #            **lm_instance_cfg["lm_eval_kwargs"], **lm_instance_cfg["vllm_kwargs"]
        #        )
        self.model_name = lm_instance_cfg['model_name']
        self.cfg = lm_instance_cfg
        self.LM = lm_eval.api.registry.get_model("vllm").create_from_arg_obj(
            lm_instance_cfg["vllm_kwargs"], lm_instance_cfg["lm_eval_kwargs"]
        )


    
@pytest.fixture(scope="module")
def lm_instance(request) -> LMInstance:
    return LMInstance(request.param)

@pytest.fixture
def task_cfg(request) -> dict:
    return request.param

@pytest.fixture(autouse=True)
def lm_task(lm_instance: LMInstance, task_cfg: dict) -> LMTask:
    return LMTask(lm_instance, task_cfg)

class LMConfigs:
    llama3_1_8b_instruct_bs128_bf16 = {
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "lm_eval_kwargs": {"batch_size": "auto"},
        "vllm_kwargs": {
            "pretrained": "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct",
            "max_num_seqs": 128,
            "max_model_len": 8192,
            "dtype": "bfloat16",
        },
    }

class TaskConfigs:
    gsm8k_llama_cot = {
        "task_name": "gsm8k_cot_llama",
        "eval_kwargs": {
            "limit": None,
            "fewshot_as_multiturn": True,
            "apply_chat_template": True,
        },
    }

    ifeval = {
        "task_name": "ifeval",
        "task_config_overrides": {"fewshot_config": {"sampler": "first_n"}},
        "eval_kwargs": {
            "limit": None,
            "fewshot_as_multiturn": True,
            "apply_chat_template": True,
        },
    }

class LMTaskTargets:
    default_atol = 0.05
    default_rtol = 0.05
    targets = {
        'Meta-Llama-3.1-8B-Instruct': {
            "gsm8k_cot_llama": {'score': 0.845},
            "ifeval": {'score': 0.804}
        }
    }

@pytest.mark.parametrize(
    "task_cfg",
    [TaskConfigs.gsm8k_llama_cot, TaskConfigs.ifeval], ids=['gsm8k_llama_cot', 'ifeval'], indirect=True
)
@pytest.mark.parametrize('lm_instance', [LMConfigs.llama3_1_8b_instruct_bs128_bf16], ids=['llama3_1_8b_instruct_bs128_bf16'], indirect=True)
def test_task(lm_task: LMTask) -> None:
    res = lm_task.run_evaluate()
    task_name = lm_task.task_name
    model_name = lm_task.lm_instance.model_name
    metrics_to_extract = [m['metric'] for m in lm_task.task_dict[lm_task.task_name]._config.metric_list] # ugh...
    score = statistics.mean(
        [v for k,v in res['results'][lm_task.task_name].items() for metric in metrics_to_extract if metric in k and "stderr" not in k] # UGH...
    )
    target_dict = LMTaskTargets.targets[model_name][task_name]
    target_score = target_dict['score']
    atol = target_dict['atol'] if 'atol' in target_dict else LMTaskTargets.default_atol
    rtol = target_dict['rtol'] if 'rtol' in target_dict else LMTaskTargets.default_rtol
    np.testing.assert_allclose(score, target_score, atol=atol, rtol=rtol)
