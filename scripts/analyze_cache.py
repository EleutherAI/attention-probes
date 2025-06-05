#%%
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

cache_dir = Path("../cache")
configs = ["v0", "v0-last"]
# %%
results = defaultdict(dict)
for config in configs:
    for run in (cache_dir / config).glob("*"):
        results[run.stem][config] = np.median(json.load(open(run / "eval_results.json"))["accuracies"])
# %%
for setup in results:
    print(setup)
    for config in results[setup]:
        print(f"    {config}: {results[setup][config]:.2f}")

# %%
