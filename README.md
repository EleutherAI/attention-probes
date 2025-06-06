# Attention probes

Based on https://github.com/shan23chen/MOSAIC (unlicensed)

## Install

`uv sync`

## Run

```
uv run scripts_mosaic/run_experiments.py --extract-only
uv run -m attention_probe --run_set v1
uv run -m attention_probe --run_set v1-last --last_only
uv run -m attention_probe --run_set v1-mean --take_mean
uv run -m attention_probe --run_set v1-tanh --use_tanh
```

## Analyze

Open `scripts/analyze_cache.py` as Jupytext and run.
