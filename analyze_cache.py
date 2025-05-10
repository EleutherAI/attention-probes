#%%
from pathlib import Path
import joblib

cache_dir = Path("cache")
cache = []
for cache_path in cache_dir.glob("*.pkl"):
    result = joblib.load(cache_path)
    cache.append(result)
#%%
import hashlib

def hash_dict(d):
    d = sorted(d.items())
    return hashlib.sha256(str(d).encode()).hexdigest()
all_configs = set()
for r in cache:
    config = r["config"]
    config_hash = hash_dict(config)
    if config_hash not in all_configs:
        print(config_hash, config)
    all_configs.add(config_hash)
selected_configs = {
    "489e9a72bb3fdc41dba4ff60c3caae3851e596f995ada8dc7067c290dc903a63":
        "attn",
    "8c6cf85ec0b1f2ae038d2baa3caf4b98ea50bafb7fa5d4389ce7d2e9d6b564a5":
        "mean",
}
# %%
from collections import defaultdict
all_datas = defaultdict(list)
for r in cache:
    data_info = r["data_info"]
    # data_key = data_info.copy()
    # del data_key["sae_size"]
    # data_hash = hash_dict(data_key)
    data_hash = (data_info["model_name"], data_info["dataset_path"], data_info["layer_num"])
    all_datas[data_hash].append(r)
# %%
import numpy as np
for data_hash, rs in sorted(all_datas.items()):
    for_model = defaultdict(list)
    for r in rs:
        config = r["config"]
        config_hash = hash_dict(config)
        if config_hash not in selected_configs:
            continue
        for_model[selected_configs[config_hash]].append(r)
    if len(for_model) != len(selected_configs):
        continue
    # di = rs[0]["data_info"]
    # print(di["model_name"], di["layer_num"], di["dataset_path"])
    # print([x["data_info"]["sae_size"] for x in rs])
    print(*data_hash)
    for k, v in sorted(for_model.items()):
        print("", k, max([np.median(r["eval_results"]["accuracies"]) for r in v]))
# %%
