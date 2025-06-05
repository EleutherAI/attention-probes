#%%
from pathlib import Path
import joblib

cache_dir = Path("cache")
cache = []
for cache_path in cache_dir.glob("*.pkl"):
    result = joblib.load(cache_path)
    # for s in result["eval_results"].get("roc_aucs") or []:
    #     if s == 0.0:
    #         cache_path.unlink()
    #         break
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
    "7d0daf31224a99363a91df7bd4ed113fac0208efe455f5bdd72025e658b0a9fe":
        "attn",
    "c6f60fab8a5678dfdb04d6f43c5e7291b27e43250e1139b67562a7e9c663b624":
        "mean",
    "8d5bf89e6590d135fb112d935010ecfd963f60da760d8cd1eacd3d68b52525d1":
        "last",
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
        print("", k, max([np.mean(r["eval_results"]["accuracies"]) for r in v]))
# %%
