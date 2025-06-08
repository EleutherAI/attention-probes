#%%
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

cache_dir = Path("../cache")
# configs = ["v0", "v0-last"]
# configs = ["v1", "v1-last", "v1-mean", "v1-tanh"]
# configs = ["v1", "v1-mean"]
configs = ["h0", "h0-last"]
# configs = ["v1", "v1-tanh"]
# configs = ["v1", "v1-last"]
results = defaultdict(dict)
for config in configs:
    for run in (cache_dir / config).glob("*"):
        try:
            results[run.stem][config] = np.mean(json.load(open(run / "eval_results.json"))["accuracies"])
        except FileNotFoundError:
            pass
xs = []
ys = []
for setup, config_results in results.items():
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

names = {
    "v1": "Attention Probe",
    "v1-last": "Last Token Probe",
    "v1-mean": "Mean Probe",
    "v1-tanh": "Tanh Probe",
    "h0": "Neurons in a Haystack Attention Probe",
    "h0-last": "Neurons in a Haystack Last Token Probe",
}
plt.plot([0, 1], [0, 1], "k--")
plt.scatter(xs, ys)
plt.xlabel(names[configs[0]])
plt.ylabel(names[configs[1]])
plt.xlim(0, 1)
plt.ylim(0, 1)
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
    htmls = json.load(open(html_path))
    for html in htmls:
        display(HTML(html))
    break
# %%
