#%%
import torch
import random
import subprocess
import os
import threading
from tqdm import tqdm
import pynvml  # For GPU memory checking

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pynvml.nvmlInit()
available_devices = set()
for i in range(torch.cuda.device_count()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if info.used / 1024**2 <= 1024:  # Convert bytes to MB and check if <= 1GB
        available_devices.add(i)
pynvml.nvmlShutdown()

print(f"Available GPUs: {available_devices}")

tasks = []
for prefix, data_source in [
    ("h-", "output"),
    ("hay-", "output_haystack"),
]:
    architectures = [
        ("mean", "--take_mean=True"),
        ("mean-adam", "--take_mean=True --train_lbfgs=False"),
        ("last", "--last_only=True"),
        ("last-adam", "--last_only=True --train_lbfgs=False"),
    ]
    for n_heads in [1, 2, 8]:
        architectures.append((f"attn-{n_heads}", f"--train_lbfgs=False --n_heads={n_heads}"))
    for arch_name, arch_args in architectures:
        for seed in [0, 100, 200]:
            tasks.append([
                "python", "-m",
                "attention_probe.train_mosaic",
                "--run_set",
                f"{prefix}{arch_name}-{seed}",
                "--cache_source", data_source,
                *arch_args.split(),
                "--seed",
                str(seed),
            ])

random.shuffle(tasks)
n_can_run_parallel = 1 
n_occupants = {gpu: 0 for gpu in available_devices}
start_end_mutex = threading.Lock()
wake_up_signal = threading.Condition(start_end_mutex)

def run_task(task):
    with start_end_mutex:
        while all(n_occupants[gpu] >= n_can_run_parallel for gpu in available_devices):
            wake_up_signal.wait()
        gpu = min(available_devices, key=lambda gpu: n_occupants[gpu])
        assert n_occupants[gpu] < n_can_run_parallel
        n_occupants[gpu] += 1

    try:
        pipes = subprocess.Popen(
            task,
            env=os.environ | dict(CUDA_VISIBLE_DEVICES=str(gpu)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = pipes.communicate()
        if pipes.returncode != 0:
            print(f"Error running task on GPU {gpu}")
            print(stdout.decode("utf-8"))
            print(stderr.decode("utf-8"))
    finally:
        with start_end_mutex:
            n_occupants[gpu] -= 1
            wake_up_signal.notify()

# Launch all tasks with threading
threads = []
for task in tasks:
    thread = threading.Thread(target=run_task, args=(task,))
    threads.append(thread)
    thread.start()

# Wait for all tasks to complete
for thread in tqdm(threads, desc="Running tasks"):
    thread.join()

#%%