#%%
from attention_probe import AttentionProbe

from torch.nn import functional as F

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
from dataclasses import dataclass

@dataclass
class AttentionProbeTrainConfig(Serializable):
    """
    Config for training an attention probe.
    """
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    train_iterations: int = 1000
    n_heads: int = 1
    last_only: bool = False
    take_mean: bool = False
    hidden_dim: int = 0
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 5

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
    multi_class: bool
    
    @torch.no_grad()
    def reindex(self, indices: Int[Array, "batch_size"]) -> "TrainingData":
        return TrainingData(
            x=self.x[indices],
            mask=self.mask[indices],
            position=self.position[indices],
            y=self.y[indices],
            input_ids=self.input_ids[indices],
            multi_class=self.multi_class,
        )
    
    @torch.no_grad()
    def split(self, n_folds: int, seed: int) -> list[tuple["TrainingData", "TrainingData"]]:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(self.x, self.y))
        return [(self.reindex(split[0]), self.reindex(split[1])) for split in splits]
    
    @torch.no_grad()
    def to_tensor(self, device: torch.device = None) -> "TrainingData":
        return TrainingData(
            x=torch.tensor(self.x, dtype=torch.float32, device=device),
            mask=torch.tensor(self.mask, dtype=torch.bool, device=device),
            position=torch.tensor(self.position, dtype=torch.int32, device=device),
            y=torch.tensor(self.y, dtype=torch.int32, device=device),
            input_ids=torch.tensor(self.input_ids, dtype=torch.int32, device=device),
            multi_class=self.multi_class,
        )
    
    @property
    def is_tensor(self) -> bool:
        return isinstance(self.x, torch.Tensor)
    
    @torch.no_grad()
    def process(self, config: MulticlassTrainConfig) -> "TrainingData":
        if not self.is_tensor:
            self = self.to_tensor(device="cpu")
        
        if config.take_mean:
            mask_sum = self.mask.sum(-1, keepdim=True)[..., None]
            mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
            self.x = self.x * 0 + (self.x * self.mask[..., None]).sum(-2, keepdim=True) / mask_sum
        self.x = torch.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        if config.last_only:
            self.mask = self.mask * (self.position == (self.position * self.mask).max(dim=-1, keepdim=True).values)
        return self

def get_data(training_data_origin: TrainingDataOrigin) -> TrainingData:
    metadata = pd.read_csv(training_data_origin.metadata_path)
    metadata = metadata.drop_duplicates(subset=['npz_file'])
    cache_data = [np.load(file) for file in metadata['npz_file']]
    x_data = [npf['hidden_state'] for npf in cache_data]
    input_ids = [npf['input_ids'] for npf in cache_data]
    max_seq_len = max([x.shape[0] for x in x_data])
    x_data_padded = np.array([np.pad(x, ((0, max_seq_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) for x in x_data])
    mask_data = np.array([np.pad(np.ones(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
    position_data = np.array([np.pad(np.arange(x.shape[0]), (0, max_seq_len - x.shape[0]), mode='constant', constant_values=0) for x in x_data])
    x_data = x_data_padded
    input_ids = np.array([np.pad(np.array(input_id), (0, max_seq_len - len(input_id)), mode='constant', constant_values=0) for input_id in input_ids])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata['label'])
    multi_class = len(np.unique(y)) > 2
    return TrainingData(
        x=x_data,
        mask=mask_data,
        position=position_data,
        y=y,
        input_ids=input_ids,
        multi_class=multi_class,
    )

@dataclass
class RunConfig(Serializable):
    train_config: MulticlassTrainConfig = field(default_factory=MulticlassTrainConfig)
    datasets: list[str] = list_field("Anthropic_election_questions", "AIM-Harvard_reject_prompts", "jackhhao_jailbreak-classification")
    models: list[str] = list_field("google/gemma-2-2b")
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
    print(args)
    config = args.train_config
    device = torch.device(config.device)
    print("Training on", device)
    
    for metadata_path in args.cache_source.glob("**/*.csv"):
        dataset_path = metadata_path.parents[3].name
        match = re.match(r"([a-zA-Z0-9\-]+)_([0-9]+)_activations_metadata", metadata_path.stem)
        if match is None:
            print("Warning: No match found for", metadata_path)
            continue
        model_name, layer_num = match.groups()
        model_name = TOKENIZER_NAMES[model_name]
        layer_num = int(layer_num)
        sae_size = metadata_path.parents[0].name
        
        if sae_size not in args.sae_sizes:
            print("Warning: SAE size not matched:", sae_size)
            continue
        if model_name not in args.models:
            print("Warning: Model not matched:", model_name)
            continue
        if dataset_path not in args.datasets:
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
        
        tokenizer = AutoTokenizer.from_pretrained(training_data_origin.model_name)
        training_data = get_data(training_data_origin)
        training_data = training_data.process(config)
        metrics = defaultdict(list)
        htmls = []
        for train_split, test_split in training_data.split(config.n_folds, config.seed):
            hidden_dim = train_split.x.shape[-1]
            probe = AttentionProbe(hidden_dim, config.n_heads, hidden_dim=config.hidden_dim, output_dim=1 if not train_split.multi_class else len(np.unique(train_split.y)))
            probe = probe.to(device, torch.float32)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            train_tensor = train_split.to_tensor(device=device)
            for _ in (bar := trange(config.train_iterations, desc=f"Training {model_name} {layer_num} {sae_size}")):
                optimizer.zero_grad()
                indices = torch.randint(0, len(train_split.y), (config.batch_size,), device=device)
                train_batch = train_tensor.reindex(indices)
                with torch.autocast(device_type=device.type):
                    out = probe(train_batch.x, train_batch.mask, train_batch.position)
                    if not train_split.multi_class:
                        loss = F.binary_cross_entropy_with_logits(out, train_batch.y.float()[..., None])
                    else:
                        loss = F.cross_entropy(out, train_batch.y)
                loss.backward()
                optimizer.step()
                bar.set_postfix(loss=loss.item())
        
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

            for i in np.random.randint(0, len(attns), 15):
                input_id, attn, label = test_data.input_ids[i], attns[i], test_data.y[i]
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
    
        eval_results = dict(
            accuracies=accuracies,
            roc_aucs=roc_aucs,
        )
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(eval_results, f)
        with open(save_path / "htmls.json", "w") as f:
            json.dump(htmls, f)
