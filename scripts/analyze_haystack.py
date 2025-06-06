#%%
import os
if "data" not in os.listdir():
    %cd ..
# %%
from matplotlib import pyplot as plt
from pathlib import Path
import datasets
for dataset in Path("data/gurnee_data_processed").glob("*"):
    ds = datasets.load_from_disk(dataset)
    print(dataset.name, len(ds))
    counts = []
    for label in ds["labels"]:
        counts.append(label.count("|"))
    plt.hist(counts, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if counts[0] == 0:
        print(ds["labels"][:10])
    plt.show()
# %%
template = '''  - name: "{}"
    config_name: ""
    split: "train"
    text_field: "text"
    label_field: "labels"'''
for dataset in Path("data/gurnee_data_processed").glob("*"):
    print(template.format("./" + str(dataset)))