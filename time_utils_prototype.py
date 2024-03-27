import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration
import asyncio
import signal
import multiprocessing
import pyhlml
from vllm.hpu.time_utils import device_utilization_counter
     
async def engine_step():
    print("Started engine_step!")
    for i in range(500):
        if i < 250:
            x = torch.zeros(100, device=torch.device("hpu"))
            x += torch.ones(100)
        device = pyhlml.hlmlDeviceGetHandleByIndex(0)
        device_utilization = pyhlml.hlmlDeviceGetTemperature(device, 1)
        print(f"\tDevice {0} | utilization: {device_utilization}")
        #await asyncio.sleep(0.5)
        print(f"Step {i}")
    return "Got req from engine step!"

async def run_engine():
    # device_utilization_queue = multiprocessing.Queue(maxsize=1)
    # device_utilization_results = multiprocessing.Manager().list()
    # device_utilization_process = multiprocessing.Process(
    #     target=device_utilization_counter,
    #     args=(device_utilization_queue, device_utilization_results)
    # )
    # device_utilization_process.start()    
    pyhlml.hlmlInit()

    engine_step_output = await engine_step()
    await asyncio.sleep(0)    

    # device_utilization_process.kill()
    # device_utilization_process.join()
    # utilization_data = device_utilization_results[-1] if device_utilization_results else None
    # print(f'HPU mean utilization = {utilization_data}')

def main():
    asyncio.run(run_engine())

if __name__ == "__main__":
    main()