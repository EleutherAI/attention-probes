# Attention probes

Based on https://github.com/shan23chen/MOSAIC (unlicensed). The original README can be found in `README_MOSAIC.md`.

## Install

```bash
# install
uv pip install git+https://github.com/EleutherAI/attention-probes.git

# development
git clone https://github.com/EleutherAI/attention-probes && cd attention-probes
uv sync && uv pip install -e .
```

## Experiments (from blog post)

### Collect data

```bash

# caches to /output
uv run scripts_mosaic/run_experiments.py --extract-only

# for Neurons in a Haystack paper data:
# step 1: download https://www.dropbox.com/scl/fo/14oxabm2eq47bkw2u0oxo/AKFTcikvAB8-GVdoBztQHxE?rlkey=u9qny1tsza6lqetzzua3jr8xn&e=1&dl=0
# step 2: unzip into data/gurnee_data
# step 3: uv run python scripts/gurnee_data.py
# step 4 (caches to /output_haystack):
# uv run scripts/run_haystack
```

### Run experiments

```bash
uv run -m attention_probe --run_set v1
uv run -m attention_probe --run_set v1-last --last_only
uv run -m attention_probe --run_set v1-mean --take_mean
uv run -m attention_probe --run_set v1-tanh --use_tanh
```

### Analyze

Open `scripts/analyze_cache.py` as Jupytext and run. Edit `configs[]` to change which experiments to compare. Update `names{}` for custom experiment names.

## API

```python
# Example usage
# Overfit an attention probe on a small dataset
from attention_probe import AttentionProbe, AttentionProbeTrainConfig, TrainingData, train_probe, evaluate_probe, compute_metrics
import torch

dataset_size = 1024
seq_len = 16
hidden_dim = 256
num_classes = 2
n_heads = 2

data = TrainingData(
    x=torch.randn(dataset_size, seq_len, hidden_dim),
    y=torch.randint(0, num_classes, (dataset_size,)),
    mask=torch.ones(dataset_size, seq_len),
    position=torch.arange(seq_len),
    n_classes=num_classes,
    class_mapping={0: "class 0", 1: "class 1"},
)

config = AttentionProbeTrainConfig(
    n_heads=n_heads,
    hidden_dim=hidden_dim,
)
probe, _loss = train_probe(data, config, device="cuda" if torch.cuda.is_available() else "cpu")
probs = evaluate_probe(probe, data, config)
metrics = compute_metrics(probs, data)
print(metrics)
```


