import statistics
from dataclasses import replace

from lm_eval import tasks, evaluator
import numpy as np
import pytest
import itertools
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMTask:

    def __init__(self, lm_instance, task_cfg):
        self.lm_instance = lm_instance
        self.task_cfg = task_cfg
        self.task_manager = tasks.TaskManager(include_path="./meta-configs")
        assert "task_name" in self.task_cfg, ("Task config must contain "
                                              "a task_name!")
        self.task_name = self.task_cfg["task_name"]
        self.task_dict = tasks.get_task_dict(self.task_name, self.task_manager)
        if "task_config_overrides" in self.task_cfg:
            self.task_dict[self.task_name]._config = replace(
                self.task_dict[self.task_name]._config,
                **self.task_cfg["task_config_overrides"])

    def patch_parallel_state(self):
        # NOTE(kzawora): This a really nasty workaround - for whatever reason,
        # tensor and pipeline parallel states are getting corrupted when moving
        # to a new test context and need to be re-initialized.
        # Possibly other vllm globals can be corrupted too.
        # Recognition to anyone who figures out how to prevent it, 
        # while maintaining vLLM instance reuse across tests.
        # For now, we restore TP & PP groups saved in lm_instance. Nasty.
        # If this makes you worried, remove vLLM instance reuse.
        # (remove scope='module' from lm_instance fixture)
        vllm = pytest.importorskip('vllm')
        if vllm.distributed.parallel_state._PP is None:
            vllm.distributed.parallel_state._PP = self.lm_instance.pp_group
            logger.warning('vLLM pipeline parallel state is empty!')
        if vllm.distributed.parallel_state._TP is None:
            vllm.distributed.parallel_state._TP = self.lm_instance.tp_group
            logger.warning('vLLM tensor parallel state is empty!')
        if vllm.distributed.parallel_state._WORLD is None:
            vllm.distributed.parallel_state._WORLD = self.lm_instance.world
            logger.warning('vLLM world state is empty!')

    def run_evaluate(self):
        self.patch_parallel_state()
        if self.task_cfg.get('eval_kwargs', None) is None:
            self.task_cfg["eval_kwargs"] = {}
        results = evaluator.evaluate(lm=self.lm_instance.LM,
                                     task_dict=self.task_dict,
                                     **self.task_cfg["eval_kwargs"])
        return results


class LMInstance:

    def __init__(self, lm_instance_cfg, vllm):
        self.model_name = lm_instance_cfg['model_name']
        self.cfg = lm_instance_cfg
        from lm_eval.models.vllm_causallms import VLLM
        self.LM = VLLM(**lm_instance_cfg["vllm_kwargs"],
                       **lm_instance_cfg["lm_eval_kwargs"])
        self.pp_group = vllm.distributed.parallel_state._PP
        self.tp_group = vllm.distributed.parallel_state._TP
        self.world = vllm.distributed.parallel_state._WORLD


#        from lm_eval import api
#        self.LM = api.registry.get_model("vllm").create_from_arg_obj(
#            lm_instance_cfg["vllm_kwargs"], lm_instance_cfg["lm_eval_kwargs"])


def assert_server_idle(lm):
    running = len(lm.model.llm_engine.scheduler[0].running)
    waiting = len(lm.model.llm_engine.scheduler[0].waiting)
    swapped = len(lm.model.llm_engine.scheduler[0].swapped)
    assert running == 0, f'There are {running} requests running!'
    assert waiting == 0, f'There are {running} requests waiting!'
    assert swapped == 0, f'There are {running} requests swapped!'


@pytest.fixture(scope='module')
def lm_instance(request):
    vllm = pytest.importorskip('vllm')
    lm = LMInstance(request.param, vllm)
    assert_server_idle(lm.LM)
    yield lm
    assert_server_idle(lm.LM)
    logger.debug('Destroying LM instance')


@pytest.fixture
def task_cfg(request) -> dict:
    return request.param


@pytest.fixture(autouse=True)
def lm_task(lm_instance: LMInstance, task_cfg: dict):
    task = LMTask(lm_instance, task_cfg)
    assert_server_idle(task.lm_instance.LM)
    yield task
    assert_server_idle(task.lm_instance.LM)
    logger.debug('Destroying task')


class LMConfigs:
    llama3_1_8b_instruct_bs128_bf16 = {
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "lm_eval_kwargs": {
            "batch_size": "auto"
        },
        "vllm_kwargs": {
            "pretrained":
            "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct",
            "max_num_seqs": 128,
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "disable_log_stats": False
        },
    }
    llama3_1_8b_bs128_bf16 = {
        "model_name": "Meta-Llama-3.1-8B",
        "lm_eval_kwargs": {
            "batch_size": "auto"
        },
        "vllm_kwargs": {
            "pretrained":
            "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct",
            "max_num_seqs": 128,
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "disable_log_stats": False
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
        "task_config_overrides": {
            "fewshot_config": {
                "sampler": "first_n"
            }
        },
        "eval_kwargs": {
            "limit": 10,
            "fewshot_as_multiturn": True,
            "apply_chat_template": True,
        },
    }

    meta_mmlu_pro_instruct = {
        "task_name": "meta_mmlu_pro_instruct",
    }

    meta_mmlu_pro_pretrain = {
        "task_name": "meta_mmlu_pro_pretrain",
    }

    meta_math_hard = {
        "task_name": "meta_math_hard",
    }

    meta_gpqa = {
        "task_name": "meta_gpqa",
    }

    meta_ifeval = {
        "task_name": "meta_ifeval",
    }

    meta_bbh = {
        "task_name": "meta_bbh",
    }


class LMTaskTargets:
    default_atol = 0.05
    default_rtol = 0.05
    targets = {
        'Meta-Llama-3.1-8B-Instruct': {
            "gsm8k_cot_llama": {
                'score': 0.845
            },
            "ifeval": {
                'score': 0.804
            },
            "meta_math_hard": {
                'score': 0.804
            },
            "meta_gpqa": {
                "score": 0.328
            },
            "meta_ifeval": {
                'score': 0.804
            },
            "meta_mmlu_pro_instruct": {
                'score': 0.47
            },
        },
        'Meta-Llama-3.1-8B': {
            "meta_mmlu_pro_pretrain": {
                'score': 0.356
            },
            "meta_bbh": {
                'score': 0.642
            },
        }
    }


def get_task_name(task_cfg):
    return task_cfg["task_name"]


@pytest.mark.parametrize('lm_instance',
                         [LMConfigs.llama3_1_8b_instruct_bs128_bf16],
                         ids=['llama3_1_8b_instruct_bs128_bf16'],
                         indirect=True)
@pytest.mark.parametrize("task_cfg", [
    TaskConfigs.gsm8k_llama_cot, TaskConfigs.ifeval,
    TaskConfigs.meta_mmlu_pro_instruct, TaskConfigs.meta_ifeval,
    TaskConfigs.meta_gpqa, TaskConfigs.meta_math_hard
],
                         ids=get_task_name,
                         indirect=True)
def test_task_instruct(lm_task: LMTask):
    generic_test_task(lm_task)


@pytest.mark.parametrize('lm_instance', [LMConfigs.llama3_1_8b_bs128_bf16],
                         ids=['llama3_1_8b_bs128_bf16'],
                         indirect=True)
@pytest.mark.parametrize(
    "task_cfg", [TaskConfigs.meta_bbh, TaskConfigs.meta_mmlu_pro_pretrain],
    ids=get_task_name,
    indirect=True)
def test_task_pretrain(lm_task: LMTask):
    generic_test_task(lm_task)


def generic_test_task(lm_task: LMTask) -> None:
    start = time.perf_counter()
    res = lm_task.run_evaluate()
    end = time.perf_counter()
    total_time = end - start
    task_name = lm_task.task_name
    model_name = lm_task.lm_instance.model_name
    metrics_to_extract = [
        m['metric']
        for m in lm_task.task_dict[lm_task.task_name]._config.metric_list
    ]  # ugh...
    extracted_metrics = {
        k: v
        for k, v in res['results'][lm_task.task_name].items()
        for metric in metrics_to_extract if metric in k and "stderr" not in k
    }  # UGH...
    score = statistics.mean(extracted_metrics.values())
    target_dict = LMTaskTargets.targets[model_name][task_name]
    target_score = target_dict['score']
    atol = target_dict[
        'atol'] if 'atol' in target_dict else LMTaskTargets.default_atol
    rtol = target_dict[
        'rtol'] if 'rtol' in target_dict else LMTaskTargets.default_rtol
    if True:
        tokenizer = lm_task.lm_instance.LM.tokenizer
        samples = res['samples'][lm_task.task_name]
        # tokenized_inputs = [tokenizer(x['doc']['prompt'])['input_ids'] 
        #   for x in samples]
        tokenized_inputs = [
            tokenizer(x['arguments'][0][0])['input_ids'] for x in samples
        ]
        tokenized_inputs_lens = [len(x) for x in tokenized_inputs]
        tokenized_outputs = [
            list(
                itertools.chain.from_iterable(
                    tokenizer(list(itertools.chain.from_iterable(
                        x['resps'])))['input_ids'])) for x in samples
        ]
        tokenized_outputs_lens = [len(x) for x in tokenized_outputs]
        report_accuracy(extracted_metrics, score, target_score, atol, rtol)
        report_performance(tokenized_inputs_lens, tokenized_outputs_lens,
                           total_time)

    np.testing.assert_allclose(score, target_score, atol=atol, rtol=rtol)


def report_accuracy(metrics, score, target, atol, rtol):
    logger.info(
        f'accuracy: {metrics}\nfinal score: {score}\n, target: {target} (atol: {atol}, rtol: {rtol})'  # noqa: G004, E501
    )


def report_performance(input_lens, output_lens, time):
    assert len(input_lens) == len(output_lens)
    context_lens = [i + o for i, o in zip(input_lens, output_lens)]
    gen_tput = sum(output_lens) / time
    logger.info(
        f'gen tput: {gen_tput:.2f} tok/s \n'  # noqa: G004
        f'input_tokens   | min: {min(input_lens)} | max: {max(input_lens)} | mean: {statistics.mean(input_lens):.2f} | stddev: {statistics.stdev(input_lens):.2f}\n'  # noqa: E501
        f'output_tokens  | min: {min(output_lens)} | max: {max(output_lens)} | mean: {statistics.mean(output_lens):.2f} | stddev: {statistics.stdev(output_lens):.2f}\n'  # noqa: E501
        f'context_length | min: {min(context_lens)} | max: {max(context_lens)} | mean: {statistics.mean(context_lens):.2f} | stddev: {statistics.stdev(context_lens):.2f}\n'  # noqa: E501
    )
