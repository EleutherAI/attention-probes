#%%
import os
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
os.environ["HF_DATASETS_CACHE"] = "data/hf_datasets"
import sys
sys.path.append("sparse-probing-paper")
from make_feature_datasets import prepare_feature_dataset
from config import ExperimentConfig, parse_dataset_args
# step 1: download https://www.dropbox.com/scl/fo/14oxabm2eq47bkw2u0oxo/AKFTcikvAB8-GVdoBztQHxE?rlkey=u9qny1tsza6lqetzzua3jr8xn&e=1&dl=0
# unzip into data/gurnee_data
from transformers import AutoTokenizer
import datasets
from pathlib import Path
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
data_path = Path("data/gurnee_data")
out_path = Path("data/gurnee_data_processed")
os.environ["FEATURE_DATASET_DIR"] = str(data_path)
for folder in data_path.glob("*.pyth.*"):
    if folder.is_dir():
        ds_name = folder.name.partition(".")[0]
        out_dir = out_path / ds_name
        if out_dir.exists():
            print("skipping", ds_name)
            continue
        print("processing", ds_name)
        exp_cfg = ExperimentConfig(
            {
                "feature_dataset": folder.name
            },
            feature_dataset_cfg=parse_dataset_args(
                folder.name,
            )
        )
        ds, feature_datasets = prepare_feature_dataset(exp_cfg)
        columns = set(ds.column_names) - {"name", "text", "tokens"}
        input_ids = ds["tokens"]

        texts = []
        token_to_spans = []
        for seq in tqdm(input_ids):
            text = tokenizer.decode(seq)
            id_tokens = tokenizer.convert_ids_to_tokens(seq)
            b = tokenizer.convert_tokens_to_string(id_tokens)
            assert text == b
            token_to_span = np.cumsum([0] + [len(tok) for tok in id_tokens])
            token_to_spans.append(token_to_span)
            texts.append(text)

        if "class" not in columns:
            # what do we need?
            # 1. mapping from token to span
            # 2. understanding of what the classes are
            # 3. mapping from sequence to present positions&labels
            present_pos_labels = defaultdict(list)
            for class_name, (indices, classes) in feature_datasets.items():
                for index, label in zip(tqdm(indices), classes):
                    present_pos_labels[index // input_ids.shape[1]].append((index % input_ids.shape[1], label))
            
        # 4. construct labels
        labels = []
        for i, (text, token_to_spans) in enumerate(zip(tqdm(texts), token_to_spans)):
            if "class" in columns:
                labels.append(ds["class"][i])
            else:
                class_name = ["POSITIONAL"]
                for pos, label in present_pos_labels.get(i, []):
                    start = token_to_spans[pos]
                    end = token_to_spans[pos + 1]
                    class_name.append(f"{start}:{end}:{label}")
                labels.append("|".join(class_name))

        ds = datasets.Dataset.from_dict({"text": texts, "labels": labels})
        ds.save_to_disk(out_dir)
        
# %%
fett