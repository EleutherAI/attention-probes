#%%
import requests
from tqdm import tqdm
import yaml
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)
dataset_infos = {"codeparrot/github-code": {"size": 115e6, "n_classes": 30}}
for ds in tqdm(config["datasets"]):
    name = ds["name"]
    if name in dataset_infos:
        continue
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={name}"
    response = requests.get(API_URL)
    dataset_size = response.json()["size"]["dataset"]["num_rows"]
    info = requests.get(f"https://datasets-server.huggingface.co/info?dataset={name}").json()
    config = info["dataset_info"]["default"]["config_name"]
    split = next(iter(info["dataset_info"]["default"]["splits"]))
    info_url = f"https://datasets-server.huggingface.co/statistics?dataset={name}&config={config}&split={split}"
    info_response = requests.get(info_url).json()
    n_classes = None
    for label_col in ["label", "type", "gender", "rating", "language"]:
        for s in info_response["statistics"]:
            if s["column_name"] != label_col:
                continue
            try:
                n_classes = s["column_statistics"]["n_unique"]
            except KeyError:
                n_classes = s["column_statistics"]["max"] - s["column_statistics"]["min"] + 1
            break
        if n_classes is not None:
            break
    name = name.replace("/", "_")
    if n_classes is None:
        raise ValueError(f"No label column found for {name}")
    dataset_infos[name] = {"size": dataset_size, "n_classes": n_classes}
#%%
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

metric_name = "auc"
metric_name = "acc"
# metric_name = "f1"
metric_nice = dict(
    auc="ROC AUC",
    acc="Accuracy",
    f1="F1",
)[metric_name]

info_column = "size"
info_column = "n_classes"

cache_dir = Path("./cache")
# configs = ["v1-mean", "v1-8heads"]
# configs = ["v1-mean", "v1-8heads-no-ft"]
# configs = ["v1-mean", "v1-mean-ensemble"]
# configs = ["h-attn", "h-sky"]
# configs = ["h-attn", "h-mean"]
# configs = ["hay-attn", "hay-attn-8"]
# configs = ["hay-attn", "hay-mean"]
# configs = ["hay-attn", "hay-last"]
# configs = ["hay-attn-8", "hay-last"]
# configs = ["h-mean-adam", "h-mean"]
# configs = ["h-mean-adam", "h-attn-8"]
# configs = ["h-mean-adam", "h-mean-cv"]
# configs = ["h-attn-8", "h-mean- cv"]
# configs = ["h-attn", "h-mean"]
# configs = ["h-attn-0", "h-mean-0"]
# configs = ["hay-attn-0", "hay-mean-0"]
# configs = ["hay-attn-0", "hay-last-0"]
# configs = ["hay-attn-8-0", "hay-last-0"]
# configs = ["hay-attn-1", "hay-last"]
configs = ["h-mean", "h-attn-8"]
# configs = ["h-mean", "h-mean-adam"]
# configs = ["hay-attn-8", "hay-last"]
# configs = ["hay-attn-8", "hay-last-adam"]
# configs = ["hay-last", "hay-last-adam"]
# configs = ["hay-mean", "hay-last"]
results = defaultdict(lambda: defaultdict(list))
all_eval_results = {}
all_data_configs = {}
for config in configs:
    for suffix in ["-0", "-100", "-200"]:
        for run in (cache_dir / (config + suffix)).glob("*"):
            try:
                try:
                    eval_results = json.load(open(run / "eval_results.json"))
                except json.JSONDecodeError:
                    continue
                all_eval_results[run.name] = eval_results
                all_data_configs[run.name] = json.load(open(run / "data.json"))
                if metric_name == "acc":
                    # results[run.name][config] = np.mean(eval_results["accuracies"])
                    results[run.name][config].append(eval_results["accuracy"])
                elif metric_name == "f1":
                    if eval_results.get("f1s", None) is None:
                        continue
                    # results[run.name][config] = np.mean(eval_results["f1s"])
                    results[run.name][config].append(eval_results["f1"])
                elif metric_name == "auc":
                    if eval_results.get("roc_aucs", None) is None:
                        continue
                    # results[run.name][config] = np.mean(eval_results["roc_aucs"])
                    results[run.name][config].append(eval_results["roc_auc"])
            except FileNotFoundError:
                pass

for run_name, config_results in results.items():
    for config, values in config_results.items():
        results[run_name][config] = np.mean(values)

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
        try:
            colors.append(dataset_infos[all_data_configs[setup]["dataset_path"]][info_column])
        except KeyError:
            pass
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
if not colors:
    colors = None
plt.scatter(xs, ys, c=colors, norm="log")
plt.xlabel(names.get(configs[0], configs[0]) + f" ({metric_nice})")
plt.ylabel(names.get(configs[1], configs[1]) + f" ({metric_nice})")
min_xy = min(min(xs), min(ys))
max_xy = max(max(xs), max(ys))
plt.xlim(min_xy, max_xy)
plt.ylim(min_xy, max_xy)
if colors is not None:
    plt.colorbar(label={"size": "Dataset size", "n_classes": "Number of classes"}[info_column], )
plt.savefig(f"plots/{metric_name}.png")
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