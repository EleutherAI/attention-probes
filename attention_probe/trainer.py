from typing import Optional
from collections import OrderedDict
import torch

from .attention_probe import AttentionProbe
from .linear_classifier import Classifier as LinearClassifier
from torch.nn import functional as F
from jaxtyping import Float, Int, Bool, Array
from tqdm.auto import trange
from sklearn.model_selection import StratifiedKFold
import numpy as np
from simple_parsing import Serializable
from dataclasses import dataclass, replace


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
