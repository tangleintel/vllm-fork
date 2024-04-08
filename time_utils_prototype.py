import torch
import asyncio
from vllm.utils import is_hpu
if is_hpu():
    import habana_frameworks.torch.core as htcore
    from vllm.hpu.time_utils import HpuTimeUtil
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def engine_step():
    print("Started engine_step!")
    for i in range(100):
        x = torch.zeros((2000,2000), device=torch.device("hpu"))
        x += torch.ones((2000,2000), device=torch.device("hpu"))
    return

async def run_engine():
    with HpuTimeUtil(interval=0.000005):
        engine_step_output = await engine_step()
        await asyncio.sleep(0)

def main():
    asyncio.run(run_engine())

if __name__ == "__main__":
    main()