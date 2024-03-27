from vllm.utils import is_hpu
import torch
import time

if is_hpu():
    import habana_frameworks.torch as htorch
    import pyhlml

def device_utilization_counter(queue, results):
    pyhlml.hlmlInit()

    interval = 0.000005 # seconds 
    device_count = pyhlml.hlmlDeviceGetCount()
    device_time = [0] * device_count
    mean_device_util = [0] * device_count
    loop_counter = [0] * device_count

    while True:
        for device_idx in range(device_count):
            device = pyhlml.hlmlDeviceGetHandleByIndex(device_idx)
            #device_utilization = pyhlml.hlmlDeviceGetUtilizationRates(device)
            device_utilization = pyhlml.hlmlDeviceGetPowerUsage(device)
            #print(f"\tDevice {device_idx} | utilization: {device_utilization}")

            loop_counter[device_idx] += 1
            time.sleep(interval)
            device_time[device_idx] += interval if device_utilization > 0 else 0        

            # iterative mean for each device
            mean_device_util[device_idx] += (device_utilization - mean_device_util[device_idx])/loop_counter[device_idx]

            output = (
                mean_device_util[device_idx], 
                device_time[device_idx],
            )
            queue.put(output)

        results.append(queue.get())

    #pyhlml.hlmlShutdown()