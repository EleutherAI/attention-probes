#%%
from collections import defaultdict
from pathlib import Path
import os


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


for data_source in ["h-", "hay-"]:
    config_names = []
    for arch in [
        "attn-1-", "attn-2-", "attn-8-",
        "last-", "last-adam-", "mean-", "mean-adam-"
    ]:
        for seed in [0, 100, 200]:
            config_names.append(f"{data_source}{arch}{seed}")


    base_dir = Path("cache")
    all_datasets = set()
    config_datasets = defaultdict(set)
    for config_name in config_names:
        config_dir = base_dir / config_name
        for dataset_dir in config_dir.iterdir():
            if not (dataset_dir / "eval_results.json").exists():
                continue
            all_datasets.add(dataset_dir.name)
            config_datasets[config_name].add(dataset_dir.name)

    completion_rates = {k: len(v) / len(all_datasets) for k, v in config_datasets.items()}
    completion_rate = sum(completion_rates.values()) / len(completion_rates)
    print(f"{data_source}: {completion_rate}")
# %%
