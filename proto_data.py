#%%
from attention_probe import AttentionProbe

from einops import rearrange
from torch.nn import functional as F

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict, OrderedDict
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
import argparse
from simple_parsing import Serializable, list_field, parse, field
from dataclasses import dataclass

@dataclass
class AttentionProbeTrainConfig(Serializable):
    """
    Config for training an attention probe.
    """
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    train_iterations: int = 2000
    n_heads: int = 1
    last_only: bool = False
    take_mean: bool = False
    nonlinear: bool = False
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 5

@dataclass
class MulticlassTrainConfig(AttentionProbeTrainConfig):
    n_folds: int = 5

@dataclass
class TrainingDataOrigin:
    metadata_path: Path
    model_name: str
    layer_num: int
    sae_size: str
    dataset_path: str

@dataclass
class RunConfig(Serializable):
    train_config: MulticlassTrainConfig = field(default_factory=MulticlassTrainConfig)
    datasets: list[str] = list_field("Anthropic/election_questions", "AIM-Harvard/reject_prompts", "jackhhao/jailbreak-classification")
    models: list[str] = list_field("google-gemma-2b", "google-gemma-2-2b")
    sae_sizes: list[str] = list_field("16k")
    cache_source: Path = Path("output")
    output_path: Path = Path("cache")
    display_now: bool = False
    run_set: str = "test"


TOKENIZER_NAMES = {
    "google-gemma-2b": "google/gemma-2b",
    "google-gemma-2-2b": "google/gemma-2-2b",
}


if __name__ == "__main__":
    args = parse(RunConfig)
    config = args.train_config
    device = torch.device(config.device) if torch.cuda.is_available() else "cpu"
    
    for metadata_path in args.cache_source.glob("**/*.csv"):
        dataset_path = metadata_path.parents[3].name
        match = re.match(r"([a-zA-Z0-9\-]+)_([0-9]+)_activations_metadata", metadata_path.stem)
        if match is None:
            print("Warning: No match found for", metadata_path)
            continue
        model_name, layer_num = match.groups()
        layer_num = int(layer_num)
        sae_size = metadata_path.parents[0].name
        
        if sae_size not in args.sae_sizes:
            continue
        if model_name not in args.models:
            continue
        if dataset_path not in args.datasets:
            continue
        
        training_data_origin = TrainingDataOrigin(
            metadata_path=metadata_path,
            model_name=model_name,
            layer_num=layer_num,
            sae_size=sae_size,
            dataset_path=dataset_path,
        )
        key_encoded = hashlib.sha256(json.dumps(str(metadata_path), config.to_dict()).encode()).hexdigest()
        save_path = args.output_path / args.run_set / key_encoded
        if save_path.exists():
            continue
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w") as f:
            json.dump(config.to_dict(), f)
        with open(save_path / "data.json", "w") as f:
            json.dump(training_data_origin, f)
        
        metadata = pd.read_csv(metadata_path)
        metadata = metadata.drop_duplicates(subset=['npz_file'])
        cache_data = [np.load(file) for file in metadata['npz_file']]
        x_data = [npf['hidden_state'] for npf in cache_data]
        input_ids = [npf['input_ids'] for npf in cache_data]
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAMES[model_name])
        max_seq_len = max([x.shape[0] for x in x_data])
        x_data = np.array([np.pad(x, ((0, max_seq_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) for x in x_data])
        mask_data = np.array([np.pad(np.ones(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
        position_data = np.array([np.pad(np.arange(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(metadata['label'])
        multi_class = len(np.unique(y)) > 2
        
        kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
        splits = list(kfold.split(x_data, y))
        
        print(metadata_path)
        exit()
    exit()
output_path = Path(args.output_path)
for metadata_path in output_path.glob("**/*.csv"):
    key_encoded = hashlib.sha256((str(metadata_path) + str(config)).encode()).hexdigest()
    cache_path = cache_dir / f"{key_encoded}.pkl"
    if cache_path.exists():
        continue
    
    # if "-2-" in model_name:
    #     continue
    if sae_size != "16k":
        continue
    print(f"Processing {model_name} layer {layer_num} {sae_size} with dataset {dataset_path}")
    # Turn labels from strings to ints
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
                from IPython.display import display, HTML
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