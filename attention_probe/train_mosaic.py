import os
import json
from functools import reduce
from collections import defaultdict
import warnings
import random
import torch
import re
from pathlib import Path

import pandas as pd
from jaxtyping import Float, Int, Array
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer
from simple_parsing import Serializable, list_field, parse, field
from dataclasses import dataclass, replace
from copy import copy
from sklearn.metrics import f1_score

from .trainer import TrainingData, MulticlassTrainConfig, train_probe_repeatedly, evaluate_probe


DEBUG = os.environ.get("DEBUG", "0") == "1"




@dataclass
class TrainingDataOrigin(Serializable):
    metadata_path: Path
    model_name: str
    layer_num: int
    sae_size: str
    dataset_path: str
    
    @property
    def id(self) -> str:
        return "-".join([self.model_name.replace("/", "-"), str(self.layer_num), self.sae_size, self.dataset_path])

def get_data(training_data_origin: TrainingDataOrigin, limit: int | None = None, seed: int | None = None) -> TrainingData:
    try:
        metadata = pd.read_csv(training_data_origin.metadata_path)
    except pd.errors.EmptyDataError:
        return None
    if "npz_file" not in metadata.columns:
        return None
    metadata = metadata.drop_duplicates(subset=['npz_file'])
    if limit is not None:
        metadata = metadata.sample(n=min(limit, len(metadata)), random_state=seed)
        
    label_encoder = LabelEncoder()
    labels = metadata['label'].to_list()
    max_count = max((l.count("|") if isinstance(l, str) else 0) for l in labels)
    if max_count > 1:
        return None
    elif max_count == 1:
        labels = [f'{l.partition("|")[0]}|{l.rpartition(":")[2]}' for l in labels]
    y = label_encoder.fit_transform(labels)
    
    cache_data = [np.load(file) for file in metadata['npz_file']]
    x_data = [npf['hidden_state'] for npf in cache_data]
    input_ids = [npf['input_ids'] for npf in cache_data]
    max_seq_len = max([x.shape[0] for x in x_data])
    x_data_padded = np.array([np.pad(x, ((0, max_seq_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) for x in x_data])
    mask_data = np.array([np.pad(np.ones(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
    position_data = np.array([np.pad(np.arange(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
    input_ids = np.array([np.pad(np.array(input_id), (0, max_seq_len - len(input_id)), mode='constant', constant_values=0) for input_id in input_ids])
    x_data = x_data_padded
    
    return TrainingData(
        x=x_data,
        mask=mask_data,
        position=position_data,
        y=y,
        input_ids=input_ids,
        n_classes=len(np.unique(y)),
        class_mapping=label_encoder.classes_,
    )

def compute_metrics(probs: Float[Array, "batch_size n_classes"], data: TrainingData) -> dict[str, float]:
    metrics = {}
    if data.multi_class:
        accuracy = accuracy_score(data.y, probs.argmax(axis=-1))
    else:
        accuracy = accuracy_score(data.y, probs > 0.5)
        f1 = f1_score(data.y, probs > 0.5)
        metrics['f1'] = f1
    metrics['accuracy'] = accuracy
    if not data.multi_class:
        try:
            roc_auc = roc_auc_score(data.y, probs)
        except ValueError:
            warnings.warn("ROC AUC produced an error")
            roc_auc = 0
        metrics['roc_auc'] = roc_auc
    return metrics

@dataclass
class RunConfig(Serializable):
    train_config: MulticlassTrainConfig = field(default_factory=MulticlassTrainConfig)
    # datasets: list[str] = list_field("Anthropic_election_questions", "AIM-Harvard_reject_prompts", "jackhhao_jailbreak-classification")
    datasets: list[str] = list_field()
    # models: list[str] = list_field("google/gemma-2-2b")
    models: list[str] = list_field()
    sae_sizes: list[str] = list_field("16k")
    # sae_sizes: list[str] = list_field()
    
    sweep_weight_decay: list[float] = list_field(0.0, 0.001, 0.01, 0.1)
    
    downsample: bool = True
    downsample_to: int = 16384
    min_per_class: int = 200
    
    cache_source: Path = Path("output")
    output_path: Path = Path("cache")
    display_now: bool = False
    run_set: str = "test"
    skip_existing: bool = True


TOKENIZER_NAMES = {
    "google-gemma-2b": "google/gemma-2b",
    "google-gemma-2-2b": "google/gemma-2-2b",
}


if __name__ == "__main__":
    args = parse(RunConfig)
    print(args)
    config_base = args.train_config
    device = torch.device(config_base.device)
    print("Training on", device)
    torch.set_grad_enabled(False)
    
    csv_options = list(args.cache_source.glob("**/*.csv"))
    random.shuffle(csv_options)
    for metadata_path in csv_options:
        config = copy(config_base)
        
        dataset_path = metadata_path.parents[3].name
        match = re.match(r"([a-zA-Z0-9\-]+)_([0-9]+)_activations_metadata", metadata_path.stem)
        if match is None:
            print("Warning: No match found for", metadata_path)
            continue
        model_name, layer_num = match.groups()
        model_name = TOKENIZER_NAMES[model_name]
        layer_num = int(layer_num)
        sae_size = metadata_path.parents[0].name
        
        if args.sae_sizes and sae_size not in args.sae_sizes:
            print("Warning: SAE size not matched:", sae_size)
            continue
        if args.models and model_name not in args.models:
            print("Warning: Model not matched:", model_name)
            continue
        if args.datasets and dataset_path not in args.datasets:
            print("Warning: Dataset not matched:", dataset_path)
            continue
        print("Training on", model_name, layer_num, sae_size, dataset_path)
        
        training_data_origin = TrainingDataOrigin(
            metadata_path=metadata_path,
            model_name=model_name,
            layer_num=layer_num,
            sae_size=sae_size,
            dataset_path=dataset_path,
        )
        key_encoded = training_data_origin.id
        save_path = args.output_path / args.run_set / key_encoded
        save_path.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and (save_path / "eval_results.json").exists():
            print("Skipping existing run")
            continue
    
        def skip():
            eval_results = save_path / "eval_results.json"
            if eval_results.exists():
                return
            eval_results.touch()
    
        with open(save_path / "config.json", "w") as f:
            json.dump(config.to_dict(), f)
        with open(save_path / "data.json", "w") as f:
            json.dump(training_data_origin.to_dict(), f)

        tokenizer = AutoTokenizer.from_pretrained(training_data_origin.model_name)
        training_data = get_data(training_data_origin, limit=args.downsample_to if args.downsample else None, seed=config.seed)
        if training_data is None:
            print("Warning: No training data found for", training_data_origin.id)
            skip()
            continue
        class_counts = np.bincount(training_data.y)
        if min(class_counts) < args.min_per_class:
            print("Warning: Not enough data for some classes")
            skip()
            continue
        config = replace(config, batch_size=min(config.batch_size, len(training_data)))
        
        training_data, testing_data = next(iter(training_data.split(5, config.seed)))
        
        if args.downsample:
            generator = np.random.default_rng(config.seed)
            training_data = training_data.reindex(generator.permutation(len(training_data))[:config.batch_size])
        try:
            next(iter(training_data.split(config.n_folds, config.seed)))
        except ValueError:
            print("Warning: Not enough data to split")
            skip()
            continue

        highest_accuracy, best_config = 0, None
        for weight_decay in args.sweep_weight_decay:
            splits = training_data.split(config.n_folds, config.seed)
            sweep_config = replace(config, weight_decay=weight_decay)
            val_metrics = defaultdict(list)
            for i, (train_split, test_split) in enumerate(splits):
                probe = train_probe_repeatedly(train_split, sweep_config, device)
                test_split = test_split.process(probe.config)
                test_data = test_split.to_tensor(device=device)
                probs = evaluate_probe(probe, test_data, config, compute_attn=False)
                new_metrics = compute_metrics(probs, test_split)
                for k, v in new_metrics.items():
                    val_metrics[k].append(v)
            val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            if val_metrics['accuracy'] > highest_accuracy:
                highest_accuracy = val_metrics['accuracy']
                best_config = sweep_config
        
        print(f"Chosen weight decay: {best_config.weight_decay}, accuracy: {highest_accuracy}")
        config = best_config
        probe = train_probe_repeatedly(training_data, config, device)
        testing_data = testing_data.process(config)
        test_data = testing_data.to_tensor(device=device)
        probs = evaluate_probe(probe, test_data, config, compute_attn=False)
        metrics = compute_metrics(probs, testing_data)
        
        # Handle different probe types
        if config.train_skyline:
            # SkylineProbe doesn't compute attention probabilities
            probs = evaluate_probe(probe, test_data, config, compute_attn=False)
            # Set default values for attention-related metrics
            entropy = 0.0
            entropy_baseline = 0.0
            metrics['entropy'] = float(entropy)
            metrics['entropy_baseline'] = float(entropy_baseline)
            html = "SkylineProbe - no attention visualization available"
        else:
            # Regular attention probe
            probs, attns = evaluate_probe(probe, test_data, config, compute_attn=True)
            entropy = -(attns * np.log(np.maximum(attns, 1e-10))).sum(axis=1).mean()
            numbers_of_elements = (attns > 0).sum(axis=1)
            entropies = -np.log(1 / numbers_of_elements)
            entropy_baseline = entropies.mean()
            print(f"Entropy: {entropy:.2f} (Baseline: {entropy_baseline:.2f})")
            metrics['entropy'] = float(entropy)
            metrics['entropy_baseline'] = float(entropy_baseline)

            htmls = []
            for i in np.random.randint(0, len(attns), 15):
                input_id, attn, label = test_data.input_ids[i], attns[i], test_data.text_label(test_data.y[i])
                html = []
                for i, (token_id, a) in enumerate(zip(input_id, attn)):
                    if token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
                        continue
                    a = float(a[0])
                    s, f = 0.2, 0.9
                    a = min(1, s + f * a)
                    html.append(f"<span style='color: rgba(1, 0, 0, {a:.2f})'>{tokenizer.decode(token_id)}</span>")
                html = f"<div style='background-color: white; padding: 10px; color: black'>Class: {label} " + "".join(html) + "</div>"
                if args.display_now:
                    from IPython.display import display, HTML
                    display(HTML(html))
                htmls.append(html)
            html = "<br>".join(htmls)
        
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(metrics, f)
        with open(save_path / "activations.html", "w") as f:
            f.write(html)