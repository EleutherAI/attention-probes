#%%
#%%
import requests
from tqdm import tqdm
import yaml
config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)
dataset_sizes = {"codeparrot/github-code": 115e6}
for ds in tqdm(config["datasets"]):
    name = ds["name"]
    if name in dataset_sizes:
        continue
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={name}"
    response = requests.get(API_URL)
    dataset_size = response.json()["size"]["dataset"]["num_rows"]
    name = name.replace("/", "_")
    dataset_sizes[name] = dataset_size
#%%
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

metric_name = "auc"
metric_name = "acc"
# metric_name = "f1"
metric_nice = dict(
    auc="ROC AUC",
    acc="Accuracy",
    f1="F1",
)[metric_name]

cache_dir = Path("./cache")
# configs = ["v1-mean", "v1-8heads"]
# configs = ["v1-mean", "v1-8heads-no-ft"]
configs = ["v1-mean", "v1-mean-ensemble"]
results = defaultdict(dict)
all_eval_results = {}
all_data_configs = {}
for config in configs:
    for run in (cache_dir / config).glob("*"):
        try:
            try:
                eval_results = json.load(open(run / "eval_results.json"))
            except json.JSONDecodeError:
                continue
            all_eval_results[run.name] = eval_results
            all_data_configs[run.name] = json.load(open(run / "data.json"))
            if metric_name == "acc":
                results[run.name][config] = np.mean(eval_results["accuracies"])
            elif metric_name == "f1":
                if eval_results.get("f1s", None) is None:
                    continue
                results[run.name][config] = np.mean(eval_results["f1s"])
            elif metric_name == "auc":
                if eval_results.get("roc_aucs", None) is None:
                    continue
                results[run.name][config] = np.mean(eval_results["roc_aucs"])
        except FileNotFoundError:
            pass
xs = []
ys = []
colors = []
# cmap = plt.get_cmap("tab20")
for i, (setup, config_results) in enumerate(results.items()):
    print(setup)
    for config in results[setup]:
        print(f"    {config}: {config_results[config]:.2f}")
    try:
        config_results[configs[0]], config_results[configs[1]]
    except KeyError:
        pass
    else:
        xs.append(config_results[configs[0]])
        ys.append(config_results[configs[1]])
        colors.append(dataset_sizes[all_data_configs[setup]["dataset_path"]])
        # colors.append(cmap(i))

names = {
    "v1": "Attention Probe",
    "v1-no-ft": "Attention Probe (not ft'd from mean probe)",
    "v1-dropout": "Attention Probe w/ dropout",
    "v1d": "Attention Probe (downsampled)",
    "v1-last": "Last Token Probe",
    "v1d-last": "Last Token Probe (downsampled)",
    "v1-lin-last": "Linear Classifier (Last Token)",
    "v1-lin-mean": "Linear Classifier (Mean)",
    "v1-mean": "Mean Probe",
    "v1-tanh": "Tanh Probe",
    "h1": "Neurons in a Haystack Attention Probe",
    "h1-last": "Neurons in a Haystack Last Token Probe",
    "h1-mean": "Neurons in a Haystack Mean Probe",
    "v1-mean-ensemble": "Attention Probe ensembled w/ mean probe",
    "v1-8heads": "Attention Probe w/ 8 heads",
    "v1-8heads-no-ft": "Attention Probe w/ 8 heads (not ft'd from mean probe)",
}
plt.plot([0, 1], [0, 1], "k--")
plt.scatter(xs, ys, c=colors, norm="log")
plt.xlabel(names.get(configs[0], configs[0]) + f" ({metric_nice})")
plt.ylabel(names.get(configs[1], configs[1]) + f" ({metric_nice})")
min_xy = min(min(xs), min(ys))
max_xy = max(max(xs), max(ys))
plt.xlim(min_xy, max_xy)
plt.ylim(min_xy, max_xy)
plt.colorbar(label="Dataset size", )
plt.show()
# %%
from IPython.display import HTML, display
html_config = "v1"
run_filter = "election"
for run in (cache_dir / html_config).glob("*"):
    if run_filter and run_filter not in run.stem:
        continue
    display(HTML(f"<h1>{run.stem}</h1>"))
    html_path = run / "htmls.json"
    if not html_path.exists():
        continue
    # htmls = json.load(open(html_path))
    # for html in htmls:
    #     display(HTML(html))
    break
# %%
