from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import joblib

METADATA_FILEPATH = os.path.join("data", "item_metadata.csv")
ITEM_FEATURES_FILEPATH = os.path.join("data", "full_item_features.csv")
DENSE_FEATURES_WRITE_PATH = os.path.join("data", "item_dense.csv")

NUM_OF_ITEMS = 927142
NUM_OF_PROPS = 157

metadata = pd.read_csv(METADATA_FILEPATH)

# split properties and save in another column
metadata["splitted_props"] = metadata["properties"].apply(lambda x: x.split("|"))

# get the set of all properties
with open(METADATA_FILEPATH) as f:
    f.readline()  # skip the header line
    all_props = set()  # to drop duplicate values
    for line in f:
        all_props.update(map(str.strip, line.split(",", 1)[1].split("|")))

all_props_as_list = sorted(list(all_props))

prop_name_to_index = {}  # maps prop name to the proper index in matrix

for i in range(len(all_props_as_list)):
    prop_name = all_props_as_list[i]
    prop_name_to_index[prop_name] = i

item_features = np.zeros((NUM_OF_ITEMS, NUM_OF_PROPS + 2))
item_features_as_dict = defaultdict(list)

current_item_index = 0

for row in tqdm(metadata[["item_id", "splitted_props"]].to_numpy()):

    item_id = row[0]
    props = row[1]

    # add item index as feature
    item_features[current_item_index][0] = current_item_index  # index feature
    item_features[current_item_index][1] = item_id  # id feature

    for prop in props:
        prop_index = prop_name_to_index[prop]
        item_features[current_item_index, prop_index + 2] = 1  # +2 for item_index and item_id
        item_features_as_dict[item_id].append(prop_index)

    current_item_index += 1

dataframe_columns = ["item_index", "item_id"] + all_props_as_list
item_features_dataframe = pd.DataFrame(item_features, columns=dataframe_columns, dtype=int)

item_features_dataframe.to_csv(ITEM_FEATURES_FILEPATH, index=False)
