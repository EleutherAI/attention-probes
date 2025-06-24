import os
import json
from typing import Optional
from collections import defaultdict, OrderedDict
import torch
import re
from pathlib import Path

from .attention_probe import AttentionProbe
from .linear_classifier import Classifier as LinearClassifier
from torch.nn import functional as F
from jaxtyping import Float, Int, Bool, Array
import pandas as pd
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
    """Learning rate for Adam."""
    train_lbfgs: bool = True
    """Use LBFGS instead of Adam."""
    early_stop_iterations: int = 100
    """Number of iterations without improvement before stopping."""
    
    weight_decay: float = 0.0
    """Weight decay coefficient."""
    train_iterations: int = 2000
    """Number of training iterations."""
    n_heads: int = 1
    """Number of attention heads."""
    attn_dropout_p: float = 0.0
    """Dropout probability for attention probes."""
    use_linear_classifier: bool = False
    """Use a linear classifier instead of an attention probe.
    Trained with cross-validation for tuning weight decay. Uses LBFGS.
    Only use with one-token datasets."""
    
    last_only: bool = False
    """Transforms the data such that only the last token is used.
    Any attention probe becomes a one-token linear classifier."""
    take_mean: bool = False
    """Transforms the data such that the mean of the hidden states across sequence length is used.
    Any attention probe becomes a one-token linear classifier."""
    absmax_pool: bool = False
    """Transforms the data such that the absolute maximum of the hidden states across sequence length is used.
    Any attention probe becomes a one-token linear classifier."""
    finetune_attn: bool = False
    """If we are training a mean probe or a last token probe, take a trained probe
    and finetune it into an attention probe."""
    ensemble_mean: bool = False
    """If we are training an attention probe, train a mean probe head independently from the attention probe."""
    
    hidden_dim: int = 0
    """Post-probe MLP hidden dimension."""
    use_tanh: bool = False
    """Use tanh instead of softmax for attention."""
    batch_size: int = 2048
    """Batch size for training."""
    always_move_to_device: bool = True
    """Even if the data doesn't fit into the GPU, try to move it there."""
    device: str = "cuda"
    """Device to use for training."""
    seed: int = 5
    """Seed for initializing the model."""
    
    retrain_threshold: float = 0.5
    """Loss level above which we change the seeed and retrain the probe."""
    retrain_n: int = 5
    """Number of times we are allowed to retrain the probe if the loss is above the threshold."""
    
    @property
    def one_token(self) -> bool:
        return self.last_only or self.take_mean or self.absmax_pool
    
    def not_one_token(self) -> "AttentionProbeTrainConfig":
        return replace(self, last_only=False, take_mean=False, absmax_pool=False)

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
    y_addition: Optional[Float[Array, "batch_size n_classes"]] = None
    
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
            indices = torch.from_numpy(indices)
        if self.is_tensor:
            indices = indices.to(self.device)
        return replace(
            self,
            x=self.x[indices],
            mask=self.mask[indices],
            position=self.position[indices],
            y=self.y[indices],
            y_addition=self.y_addition[indices] if self.y_addition is not None else None,
            input_ids=self.input_ids[indices],
        )
    
    @torch.no_grad()
    def split(self, n_folds: int, seed: int) -> list[tuple["TrainingData", "TrainingData"]]:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(self.y, self.y))
        for train_indices, test_indices in splits:
            yield self.reindex(train_indices), self.reindex(test_indices)
    
    @torch.no_grad()
    def to_tensor(self, device: torch.device = None) -> "TrainingData":
        if self.is_tensor:
            return replace(
                self,
                x=self.x.to(device),
                mask=self.mask.to(device),
                position=self.position.to(device),
                y=self.y.to(device),
                y_addition=self.y_addition.to(device) if self.y_addition is not None else None,
                input_ids=self.input_ids.to(device),
            )
        return replace(
            self,
            x=torch.tensor(self.x, dtype=torch.float32, device=device),
            mask=torch.tensor(self.mask, dtype=torch.bool, device=device),
            position=torch.tensor(self.position, dtype=torch.int32, device=device),
            y=torch.tensor(self.y, dtype=torch.int32, device=device),
            y_addition=torch.tensor(self.y_addition, dtype=torch.float32, device=device) if self.y_addition is not None else None,
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
        if config.last_only or config.take_mean:
            mask_sum = self.mask.sum(-1, keepdim=True)[..., None]
            mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
            self.x = (self.x * self.mask[..., None]).sum(-2, keepdim=True) / mask_sum
            self.mask = torch.ones_like(self.x[..., 0])
        if config.absmax_pool:
            self.x = self.x.abs().max(dim=-1, keepdim=True).values
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
    if config.one_token:
        config = replace(config, attn_dropout_p=0.0)
    probe = AttentionProbe(
        train_split.x.shape[-1],
        config.n_heads,
        hidden_dim=config.hidden_dim,
        output_dim=1 if not train_split.multi_class else train_split.n_classes,
        use_tanh=config.use_tanh,
        attn_dropout_p=config.attn_dropout_p,
        config=config,
    )
    if config.last_only:
        probe.position_weight.data[:] = 1.0
    return probe


def train_probe_iteration(train_split: TrainingData, config: MulticlassTrainConfig, device: torch.device, start_from: AttentionProbe | None = None) -> tuple[AttentionProbe, float]:
    train_split = train_split.trim()
    probe = attention_probe_from_config(config, train_split)
    if start_from is not None:
        probe.load_state_dict(start_from.state_dict())
        torch.nn.init.normal_(probe.q.weight.data, mean=0.0, std=0.01)
        torch.nn.init.normal_(probe.position_weight.data, mean=0.0, std=0.01)
    probe = probe.to(device, torch.float32)
    if config.train_lbfgs:
        optimizer = torch.optim.LBFGS(probe.parameters(), line_search_fn="strong_wolfe", max_iter=config.train_iterations)
    else:
        optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if train_split.numel_base() < 100_000:
        train_tensor = train_split.to_tensor(device=device)
    else:
        train_tensor = train_split.to_tensor(device="cpu")
    if config.always_move_to_device:
        train_tensor = train_tensor.trim().to_tensor(device=device)
    
    loss = torch.inf
    bar = trange(config.train_iterations, desc=f"Training")
    loss_history = []
    stopped = False
    def closure():
        nonlocal loss, stopped
        optimizer.zero_grad()
        if len(train_split.y) <= config.batch_size:
            train_batch = train_tensor
        else:
            indices = torch.randint(0, len(train_split.y), (config.batch_size,), device=train_tensor.device)
            train_batch = train_tensor.reindex(indices)
        train_batch = train_batch.trim().to_tensor(device=device)
        out = probe(train_batch.x, train_batch.mask, train_batch.position)
        if train_split.y_addition is not None:
            out = out + train_split.y_addition
        if not train_split.multi_class:
            loss = F.binary_cross_entropy_with_logits(out, train_batch.y.float()[..., None])
        else:
            loss = F.cross_entropy(out, train_batch.y.long())
        train_loss = loss
        if config.train_lbfgs:
            for param in probe.parameters():
                train_loss += param.pow(2).mul(config.weight_decay).div(2).sum()
        train_loss.backward()
        bar.set_postfix(loss=loss.item())
        loss_history.append(loss.item())
        if not config.train_lbfgs and len(loss_history) > config.early_stop_iterations * 2:
            before, after = loss_history[:-config.early_stop_iterations], loss_history[-config.early_stop_iterations:]
            if min(before) < min(after):
                stopped = True
                return
        if config.train_lbfgs:
            bar.update(1)
        return loss
    
    with torch.set_grad_enabled(True):
        if config.train_lbfgs:
            optimizer.step(closure)
        else:
            for _ in bar:
                optimizer.step(closure)
                if stopped:
                    break
    return probe, loss.item()

def train_probe(train_split: TrainingData, config: MulticlassTrainConfig, device: torch.device):
    train_split_original = train_split
    train_split = train_split.process(config)
    if config.use_linear_classifier:
        assert config.one_token
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
    
    probe_start = None
    if config.finetune_attn:
        probe_start, _ = train_probe_iteration(train_split, config, device)
        config = replace(config, train_lbfgs=False).not_one_token()
        train_split = train_split_original.process(config)
    probe, loss = train_probe_iteration(train_split, config, device, start_from=probe_start)
    if loss > config.retrain_threshold and config.retrain_n > 0:
        del probe, loss
        return train_probe(train_split, replace(config, retrain_n=config.retrain_n - 1, seed=config.seed + 1), device)
    if config.ensemble_mean:
        train_predictions = evaluate_probe(probe, train_split, config)
        mean_config = replace(config, train_lbfgs=True, take_mean=True)
        train_predictions = torch.from_numpy(train_predictions).to(device)
        train_split = replace(train_split.process(mean_config), y_addition=train_predictions)
        mean_probe, _ = train_probe_iteration(train_split, mean_config, device)
        mean_probe_sd = mean_probe.state_dict()
        for name, param in probe.named_parameters():
            param.data = torch.cat([param.data, mean_probe_sd[name].data], dim=0)
        probe.n_heads += mean_probe.n_heads
    return probe

def evaluate_probe(probe: AttentionProbe, test_data: TrainingData, config: MulticlassTrainConfig, compute_attn: bool = False, use_activation: bool = False) -> \
    tuple[Float[Array, "batch_size n_heads seq_len"], Float[Array, "batch_size n_classes"]] | Float[Array, "batch_size n_classes"]:
    device = next(probe.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type=device.type):
        attns = []
        all_probs = []
        if compute_attn:
            probe.attn_hook.register_forward_hook(lambda _, __, output: attns.append(output.detach().cpu().numpy()))
        for batch_start in trange(0, len(test_data), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(test_data))
            batch = test_data.reindex(torch.arange(batch_start, batch_end, device=device))
            batch = batch.to_tensor(device=device)
            out = probe(batch.x, batch.mask, batch.position)
            if use_activation:
                if not test_data.multi_class:
                    out = out.sigmoid()[..., 0]
                else:
                    out = out.softmax(dim=-1)
            probs = out.detach().cpu().numpy()
            all_probs.append(probs)
        probe.attn_hook._foward_hooks = OrderedDict()
        if compute_attn:
            attns = np.concatenate(attns)
        probs = np.concatenate(all_probs)
    if compute_attn:
        return probs, attns
    return probs

@dataclass
class RunConfig(Serializable):
    train_config: MulticlassTrainConfig = field(default_factory=MulticlassTrainConfig)
    # datasets: list[str] = list_field("Anthropic_election_questions", "AIM-Harvard_reject_prompts", "jackhhao_jailbreak-classification")
    datasets: list[str] = list_field()
    # models: list[str] = list_field("google/gemma-2-2b")
    models: list[str] = list_field()
    sae_sizes: list[str] = list_field("16k")
    # sae_sizes: list[str] = list_field()
    
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
        if args.downsample:
            generator = np.random.default_rng(config.seed)
            training_data = training_data.reindex(generator.permutation(len(training_data))[:config.batch_size])
        metrics = defaultdict(list)
        htmls = []
        try:
            splits = training_data.split(config.n_folds, config.seed)
        except ValueError:
            print("Warning: Not enough data to split")
            skip()
            continue

        for i, (train_split, test_split) in enumerate(splits):
            with torch.set_grad_enabled(True):
                probe = train_probe(train_split, config, device)
            
            test_split = test_split.process(probe.config)
            
            new_config_path = save_path / f"config_{i}.json"
            with open(new_config_path, "w") as f:
                json.dump(probe.config.to_dict(), f)
            n_attempts += 1
            n_failures += int(config.retrain_n - probe.config.retrain_n)

            probe.eval()
            probe.requires_grad_(False)
        
            print("Evaluating")
            test_data = test_split.to_tensor(device=device)
            probs, attns = evaluate_probe(probe, test_data, config, compute_attn=True)
            entropy = -(attns * np.log(np.maximum(attns, 1e-10))).sum(axis=1).mean()
            numbers_of_elements = (attns > 0).sum(axis=1)
            entropies = -np.log(1 / numbers_of_elements)
            entropy_baseline = entropies.mean()
            print(f"Entropy: {entropy:.2f} (Baseline: {entropy_baseline:.2f})")
            metrics['entropy'].append(float(entropy))
            metrics['entropy_baseline'].append(float(entropy_baseline))

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
            entropy=metrics['entropy'],
            entropy_baseline=metrics['entropy_baseline'],
        )
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(eval_results, f)
        with open(save_path / "htmls.json", "w") as f:
            json.dump(htmls, f)
    print(f"Failed {n_failures} out of {n_attempts} attempts")
