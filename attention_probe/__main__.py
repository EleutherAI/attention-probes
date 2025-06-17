#%%
from .attention_probe import AttentionProbe
from .linear_classifier import Classifier as LinearClassifier
from torch.nn import functional as F
import os

from jaxtyping import Float, Int, Bool, Array
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict, OrderedDict
import torch
import hashlib
import re
from tqdm.auto import trange
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer
from simple_parsing import Serializable, list_field, parse, field
from dataclasses import dataclass, replace
from copy import copy
from sklearn.metrics import f1_score


DEBUG = os.environ.get("DEBUG", "0") == "1"


@dataclass
class AttentionProbeTrainConfig(Serializable):
    """
    Config for training an attention probe.
    """
    learning_rate: float = 1e-4
    train_lbfgs: bool = False
    weight_decay: float = 0.0
    train_iterations: int = 2000
    n_heads: int = 1
    use_linear_classifier: bool = False
    last_only: bool = False
    take_mean: bool = False
    hidden_dim: int = 0
    use_tanh: bool = False
    batch_size: int = 2048
    device: str = "cuda"
    seed: int = 5
    retrain_threshold: float = 0.5
    retrain_n: int = 5

@dataclass
class MulticlassTrainConfig(AttentionProbeTrainConfig):
    n_folds: int = 5

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

@dataclass
class TrainingData:
    x: Float[Array, "batch_size seq_len hidden_dim"]
    mask: Bool[Array, "batch_size seq_len"]
    position: Int[Array, "batch_size seq_len"]
    y: Int[Array, "batch_size"]
    input_ids: Int[Array, "batch_size seq_len"]
    n_classes: int
    class_mapping: dict[int, str] | None = None
    
    @property
    def multi_class(self) -> bool:
        return self.n_classes > 2
    
    def text_label(self, y: int) -> str:
        if self.class_mapping is None:
            return str(y)
        return self.class_mapping[y]
    
    @property
    def device(self) -> torch.device:
        return self.x.device

    def __len__(self) -> int:
        return len(self.x)
    
    def numel(self) -> int:
        return int(self.mask.sum())
    
    def numel_base(self) -> int:
        return self.input_ids.numel()
    
    def trim(self) -> "TrainingData":
        last_position = self.mask.any(dim=0).nonzero().tolist()[-1][-1] + 1
        return replace(
            self,
            x=self.x[:, :last_position],
            mask=self.mask[:, :last_position],
            position=self.position[:, :last_position],
            y=self.y[:, :last_position] if self.y.ndim > 1 else self.y,
            input_ids=self.input_ids[:, :last_position],
        )
    @torch.no_grad()
    def reindex(self, indices: Int[Array, "batch_size"]) -> "TrainingData":
        assert indices.ndim == 1
        assert int(max(indices)) < len(self)
        if self.is_tensor and isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(self.device)
        return replace(
            self,
            x=self.x[indices],
            mask=self.mask[indices],
            position=self.position[indices],
            y=self.y[indices],
            input_ids=self.input_ids[indices],
        )
    
    @torch.no_grad()
    def split(self, n_folds: int, seed: int) -> list[tuple["TrainingData", "TrainingData"]]:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(self.y, self.y))
        return [(self.reindex(split[0]), self.reindex(split[1])) for split in splits]
    
    @torch.no_grad()
    def to_tensor(self, device: torch.device = None) -> "TrainingData":
        if self.is_tensor:
            return replace(
                self,
                x=self.x.to(device),
                mask=self.mask.to(device),
                position=self.position.to(device),
                y=self.y.to(device),
                input_ids=self.input_ids.to(device),
            )
        return replace(
            self,
            x=torch.tensor(self.x, dtype=torch.float32, device=device),
            mask=torch.tensor(self.mask, dtype=torch.bool, device=device),
            position=torch.tensor(self.position, dtype=torch.int32, device=device),
            y=torch.tensor(self.y, dtype=torch.int32, device=device),
            input_ids=torch.tensor(self.input_ids, dtype=torch.int32, device=device),
        )
    
    @property
    def is_tensor(self) -> bool:
        return isinstance(self.x, torch.Tensor)
    
    @torch.no_grad()
    def process(self, config: MulticlassTrainConfig) -> "TrainingData":
        if not self.is_tensor:
            self = self.to_tensor(device="cpu")
        
        if config.last_only:
            self.mask = self.mask * (self.position == (self.position * self.mask).max(dim=-1, keepdim=True).values)
        if config.take_mean or config.last_only:
            mask_sum = self.mask.sum(-1, keepdim=True)[..., None]
            mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
            self.x = (self.x * self.mask[..., None]).sum(-2, keepdim=True) / mask_sum
            self.mask = torch.ones_like(self.x[..., 0])
        self.x = torch.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        return self

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


def attention_probe_from_config(config: MulticlassTrainConfig, train_split: TrainingData) -> AttentionProbe:
    return AttentionProbe(
        train_split.x.shape[-1],
        config.n_heads,
        hidden_dim=config.hidden_dim,
        output_dim=1 if not train_split.multi_class else train_split.n_classes,
        use_tanh=config.use_tanh,
        config=config,
    )


def train_probe_iteration(train_split: TrainingData, config: MulticlassTrainConfig, device: torch.device) -> tuple[AttentionProbe, float]:
    train_split = train_split.trim()
    probe = attention_probe_from_config(config, train_split)
    probe = probe.to(device, torch.float32)
    if config.train_lbfgs:
        optimizer = torch.optim.LBFGS(probe.parameters(), line_search_fn="strong_wolfe", max_iter=config.train_iterations)
    else:
        optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if train_split.numel_base() < 100_000:
        train_tensor = train_split.to_tensor(device=device)
    else:
        train_tensor = train_split.to_tensor(device="cpu")
    
    loss = torch.inf
    bar = trange(config.train_iterations, desc=f"Training")
    def closure():
        nonlocal loss
        optimizer.zero_grad()
        if len(train_split.y) <= config.batch_size:
            train_batch = train_tensor
        else:
            indices = torch.randint(0, len(train_split.y), (config.batch_size,), device=train_tensor.device)
            train_batch = train_tensor.reindex(indices)
        train_batch = train_batch.trim().to_tensor(device=device)
        out = probe(train_batch.x, train_batch.mask, train_batch.position)
        if not train_split.multi_class:
            loss = F.binary_cross_entropy_with_logits(out, train_batch.y.float()[..., None])
        else:
            loss = F.cross_entropy(out, train_batch.y.long())
        loss.backward()
        bar.set_postfix(loss=loss.item())
        if config.train_lbfgs:
            bar.update(1)
        return loss
    
    with torch.set_grad_enabled(True):
        if config.train_lbfgs:
            optimizer.step(closure)
        else:
            for _ in bar:
                optimizer.step(closure)
    return probe, loss.item()

def train_probe(train_split: TrainingData, config: MulticlassTrainConfig, device: torch.device):
    if config.use_linear_classifier:
        assert config.last_only or config.take_mean
        n_classes = train_split.n_classes
        if n_classes == 2:
            n_classes = 1
        probe = LinearClassifier(train_split.x.shape[-1], num_classes=n_classes, device=device)
        batch = train_split.to_tensor(device=device)
        x = (batch.x * batch.mask[..., None]).sum(-2)
        y = batch.y.long()
        probe.fit_cv(x, y)
        att_probe = attention_probe_from_config(config, train_split).to(device)
        att_probe.v.weight.data[:] = probe.linear.weight.data
        att_probe.v.bias.data[:] = probe.linear.bias.data
        return att_probe
    
    probe, loss = train_probe_iteration(train_split, config, device)
    if loss > config.retrain_threshold and config.retrain_n > 0:
        del probe, loss
        return train_probe(train_split, replace(config, retrain_n=config.retrain_n - 1, seed=config.seed + 1), device)
    return probe

@dataclass
class RunConfig(Serializable):
    train_config: MulticlassTrainConfig = field(default_factory=MulticlassTrainConfig)
    # datasets: list[str] = list_field("Anthropic_election_questions", "AIM-Harvard_reject_prompts", "jackhhao_jailbreak-classification")
    datasets: list[str] = list_field()
    # models: list[str] = list_field("google/gemma-2-2b")
    models: list[str] = list_field()
    # sae_sizes: list[str] = list_field("16k")
    sae_sizes: list[str] = list_field()
    cache_source: Path = Path("output")
    output_path: Path = Path("cache")
    display_now: bool = False
    run_set: str = "test"
    skip_existing: bool = False
    # Reduce dataset size to batch size
    downsample: bool = False


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
    
    n_failures, n_attempts = 0, 0
    
    for metadata_path in args.cache_source.glob("**/*.csv"):
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
        with open(save_path / "config.json", "w") as f:
            json.dump(config.to_dict(), f)
        with open(save_path / "data.json", "w") as f:
            json.dump(training_data_origin.to_dict(), f)
        if args.skip_existing and (save_path / "eval_results.json").exists():
            print("Skipping existing run")
            continue

        tokenizer = AutoTokenizer.from_pretrained(training_data_origin.model_name)
        training_data = get_data(training_data_origin, limit=config.batch_size if args.downsample else None, seed=config.seed)
        if training_data is None:
            print("Warning: No training data found for", training_data_origin.id)
            continue
        config = replace(config, batch_size=min(config.batch_size, len(training_data)))
        if args.downsample:
            generator = np.random.default_rng(config.seed)
            training_data = training_data.reindex(generator.permutation(len(training_data))[:config.batch_size])
        training_data = training_data.process(config)
        metrics = defaultdict(list)
        htmls = []
        try:
            splits = training_data.split(config.n_folds, config.seed)
        except ValueError:
            print("Warning: Not enough data to split")
            continue

        for i, (train_split, test_split) in enumerate(splits):
            with torch.set_grad_enabled(True):
                probe = train_probe(train_split, config, device)
            
            new_config_path = save_path / f"config_{i}.json"
            with open(new_config_path, "w") as f:
                json.dump(probe.config.to_dict(), f)
            n_attempts += 1
            n_failures += int(config.retrain_n - probe.config.retrain_n)

            probe.eval()
            probe.requires_grad_(False)
        
            test_data = test_split.to_tensor(device=device)
            with torch.inference_mode(), torch.autocast(device_type=device.type):
                attns = []
                probe.attn_hook.register_forward_hook(lambda _, __, output: attns.append(output.detach().cpu().numpy()))
                out = probe(test_data.x, test_data.mask, test_data.position)
                if not test_data.multi_class:
                    probs = out.sigmoid().detach().cpu().numpy()[..., 0]
                else:
                    probs = out.softmax(dim=-1).detach().cpu().numpy()
                probe.attn_hook._foward_hooks = OrderedDict()
                attns = np.concatenate(attns)

            print("Evaluating")
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
        
            if test_data.multi_class:
                accuracy = accuracy_score(test_split.y, probs.argmax(axis=-1))
            else:
                accuracy = accuracy_score(test_split.y, probs > 0.5)
                f1 = f1_score(test_split.y, probs > 0.5)
                metrics['f1'].append(f1)
            metrics['accuracy'].append(accuracy)
            if not test_data.multi_class:
                try:
                    roc_auc = roc_auc_score(test_split.y, probs)
                except ValueError:
                    print("Warning: ROC AUC produced an error")
                    roc_auc = 0
                print(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
                metrics['roc_auc'].append(roc_auc)
            else:
                print(f"Accuracy: {accuracy:.4f}")
        if not test_data.multi_class:
            roc_aucs = metrics['roc_auc']
            print(f"Mean ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
        else:
            roc_aucs = None
        accuracies = metrics['accuracy']
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        if "f1" in metrics:
            f1s = metrics['f1']
            print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        else:
            f1s = None
    
        eval_results = dict(
            accuracies=accuracies,
            roc_aucs=roc_aucs,
            f1s=f1s,
        )
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(eval_results, f)
        with open(save_path / "htmls.json", "w") as f:
            json.dump(htmls, f)
    print(f"Failed {n_failures} out of {n_attempts} attempts")
