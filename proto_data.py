#%%
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch


class AttentionProbe(nn.Module):
    def __init__(self, d_in, n_heads, output_dim: int = 1, hidden_dim: int = 0):
        super().__init__()
        self.q = nn.Linear(d_in, n_heads, bias=False)
        self.v = nn.Linear(d_in, n_heads * (hidden_dim or output_dim))
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.position_weight = nn.Parameter(torch.zeros((n_heads,), dtype=torch.float32))
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.o = nn.Linear(hidden_dim, output_dim)
        self.attn_hook = nn.Identity()
    def forward(self, x, mask, position):
        k = self.q(x) - ((1 - mask.float()) * 1e9)[..., None] + position[..., None] * self.position_weight
        p = torch.nn.functional.softmax(k, dim=-2)
        self.attn_hook(p)
        v = self.v(x).unflatten(-1, (self.n_heads, -1))
        o = (p[..., None] * v).sum((-2, -3))
        if self.hidden_dim:
            o = self.o(o.relu())
        return o
#%%
from pathlib import Path
import pandas as pd
from collections import defaultdict, OrderedDict
from IPython.display import display, HTML
import torch
import hashlib
import joblib
import re
from tqdm.auto import tqdm, trange
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer


cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
output_path = Path("output")
train_iterations = 2000
lr = 1e-4
# wd = 1e-4
wd = 0.0
display_now = False
# n_heads, last_only, take_mean, nonlinear = 1, False, False, False
n_heads, last_only, take_mean, nonlinear = 1, True, False, True
# n_heads, last_only, take_mean, nonlinear = 1, True, True, True
batch_size = 256
device = "cuda:0"
seed = 5
n_folds = 5
torch.manual_seed(seed)
device = torch.device(device) if torch.cuda.is_available() else "cpu"
tokenizer_names = {
    "google-gemma-2b": "google/gemma-2b",
    "google-gemma-2-2b": "google/gemma-2-2b",
}
for metadata_path in output_path.glob("**/*.csv"):
    config = dict(
        train_iterations=train_iterations,
        lr=lr,
        wd=wd,
        display_now=display_now,
        n_heads=n_heads,
        last_only=last_only,
        take_mean=take_mean,
        seed=seed,
        n_folds=n_folds,
    )
    key_encoded = hashlib.sha256((str(metadata_path) + str(config)).encode()).hexdigest()
    cache_path = cache_dir / f"{key_encoded}.pkl"
    if cache_path.exists():
        continue
    
    dataset_path = metadata_path.parents[3].name
    match = re.match(r"([a-zA-Z0-9\-]+)_([0-9]+)_activations_metadata", metadata_path.stem)
    if match is None:
        print("Warning: No match found for", metadata_path)
        continue
    model_name, layer_num = match.groups()
    # if "-2-" in model_name:
    #     continue
    layer_num = int(layer_num)
    sae_size = metadata_path.parents[0].name
    if sae_size != "16k":
        continue
    print(f"Processing {model_name} layer {layer_num} {sae_size} with dataset {dataset_path}")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.drop_duplicates(subset=['npz_file'])
    # Read all numpy files into an X array
    numpys = [np.load(file) for file in metadata['npz_file']]
    X = [npf['hidden_state'] for npf in numpys]
    # input_ids = [npf['input_ids'][:len(x)] for npf, x in zip(numpys, X)]
    input_ids = [npf['input_ids'] for npf, x in zip(numpys, X)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_names[model_name])
    hidden_dim = X[0].shape[-1]
    max_seq_len = max([x.shape[0] for x in X])
    # Pad sequences to the same length
    X = dict(
        x=np.array([np.pad(x, ((0, max_seq_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) for x in X]),
        mask=np.array([np.pad(np.ones(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in X]),
        position=np.array([np.pad(np.arange(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in X]),
    )

    # Turn labels from strings to ints
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata['label'])
    multi_class = len(np.unique(y)) > 2
    # if not multi_class:
    #     continue

    # Create cross-validation splits
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(y, y))
    
    metrics = defaultdict(list)
    for train, test in splits:
        train_y, test_y = torch.tensor(y[train]).to(device), y[test]
        train_x = {k: torch.tensor(v[train]).to(device) for k, v in X.items()}
        test_x = {k: torch.tensor(v[test]).to(device) for k, v in X.items()}
        
        probe = AttentionProbe(hidden_dim, n_heads, hidden_dim=128 if nonlinear else 0, output_dim=1 if not multi_class else len(np.unique(y)))
        probe = probe.to(device, torch.float32)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        
        for _ in (bar := trange(train_iterations, desc=f"Training {model_name} {layer_num} {sae_size}")):
            optimizer.zero_grad()
            indices = torch.randint(0, len(train_y), (batch_size,))
            batch = {k: torch.nan_to_num(v[indices], nan=0.0, posinf=0.0, neginf=0.0) for k, v in train_x.items()}
            mask = batch['mask'].float()
            position = batch['position']

            if take_mean:
                mask_sum = mask.sum(-1, keepdim=True)[..., None]
                mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
                batch['x'] = batch['x'] * 0 + batch['x'].sum(-2, keepdim=True) / mask_sum
            batch['x'] = torch.nan_to_num(batch['x'], nan=0.0, posinf=0.0, neginf=0.0)
            if last_only:
                mask = mask * (position == position.max(axis=-1, keepdims=True).values)
            with torch.autocast(device_type=device.type):
                out = probe(batch['x'], mask, position)
                if not multi_class:
                    loss = F.binary_cross_entropy_with_logits(out, train_y[indices].float()[..., None])
                else:
                    loss = F.cross_entropy(out, train_y[indices])
                loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())
        if loss.item() == float("nan"):
            print("Warning: Loss is NaN")
            continue
        
        with torch.inference_mode(), torch.autocast(device_type=device.type):
            attns = []
            probe.attn_hook.register_forward_hook(lambda _, __, output: attns.append(output.detach().cpu().numpy()))
            out = probe(test_x['x'], test_x['mask'], test_x['position'])
            if not multi_class:
                probs = out.sigmoid().detach().cpu().numpy()[..., 0]
            else:
                probs = out.softmax(dim=-1).detach().cpu().numpy()
            probe.attn_hook._foward_hooks = OrderedDict()
            attns = np.concatenate(attns)
        

        htmls = []
        for i in np.random.randint(0, len(attns), 5):
            input_id, attn, label = input_ids[test[i]], attns[i], metadata["label"].iloc[test[i]]
            html = []
            for i, (token_id, a) in enumerate(zip(input_id, attn)):
                if token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
                    continue
                a = float(a[0])
                s, f = 0.2, 0.9
                a = min(1, s + f * a)
                html.append(f"<span style='color: rgba(1, 0, 0, {a:.2f})'>{tokenizer.decode(token_id)}</span>")
            html = f"<div style='background-color: white; padding: 10px; color: black'>Class: {label} " + "".join(html) + "</div>"
            if display_now:
                display(HTML(html))
            htmls.append(html)
        
        if multi_class:
            accuracy = accuracy_score(test_y, probs.argmax(axis=-1))
        else:
            accuracy = accuracy_score(test_y, probs > 0.5)
        metrics['accuracy'].append(accuracy)
        if not multi_class:
            try:
                roc_auc = roc_auc_score(test_y, probs)
            except ValueError:
                print("Warning: ROC AUC produced an error")
                roc_auc = 0
            print(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
            metrics['roc_auc'].append(roc_auc)
        else:
            print(f"Accuracy: {accuracy:.4f}")
    if not multi_class:
        roc_aucs = metrics['roc_auc']
        print(f"Mean ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    else:
        roc_aucs = None
    accuracies = metrics['accuracy']
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    result = dict(
        data_info=dict(
            metadata_path=metadata_path,
            model_name=model_name,
            layer_num=layer_num,
            sae_size=sae_size,
            dataset_path=dataset_path,
        ),
        eval_results=dict(
            accuracies=accuracies,
            roc_aucs=roc_aucs,
        ),
        htmls=htmls,
        config=config
    )
    joblib.dump(result, cache_path)
#%%