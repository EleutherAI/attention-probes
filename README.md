# Attention probes

Based on https://github.com/shan23chen/MOSAIC (unlicensed)

## Install

`uv sync`

## Run

```
# caches to /output
uv run scripts_mosaic/run_experiments.py --extract-only

# for Neurons in a Haystack paper data:
# step 1: download https://www.dropbox.com/scl/fo/14oxabm2eq47bkw2u0oxo/AKFTcikvAB8-GVdoBztQHxE?rlkey=u9qny1tsza6lqetzzua3jr8xn&e=1&dl=0
# step 2: unzip into data/gurnee_data
# step 3: uv run python scripts/gurnee_data.py
# step 4 (caches to /output_haystack):
# uv run scripts/run_haystack

# run experiments
uv run -m attention_probe --run_set v1
uv run -m attention_probe --run_set v1-last --last_only
uv run -m attention_probe --run_set v1-mean --take_mean
uv run -m attention_probe --run_set v1-tanh --use_tanh
```

## Analyze

Open `scripts/analyze_cache.py` as Jupytext and run. Edit configs[] to change which experiments to compare. Update names{} for custom experiment names.
