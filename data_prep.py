import pandas as pd
import numpy as np
import arrow
import os
from helpers import get_dataframe_memory_usage

TRAIN_DATA_PATH = os.path.join("data","train.csv")
TEST_DATA_PATH = os.path.join("data", "test.csv")
METADATA_FILEPATH = os.path.join("data","item_metadata.csv")
SAMPLED_DATA_PATH = os.path.join("data", "sampled", "train_sample.csv")
EVENTS_WRITE_PATH = os.path.join("data", "events_sorted.csv")

train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

assert np.all(train_data.columns == test_data.columns)

events = pd.read_csv(SAMPLED_DATA_PATH, index_col=0)

# DATA PREPARATION ###
# This part is inspired from the Logic AI, Layer6 AI and PVZ teams' solutions to the challenge.

events["src"] = "train"
events["is_test"] = 0

events.sort_values(["timestamp", "user_id", "step"], inplace=True)

# fill empty impressions with backward filling, impressions carried to earlier steps
events["fake_impressions"] = events.groupby(["user_id", "session_id"])["impressions"].bfill()
events["fake_prices"] = events.groupby(["user_id", "session_id"])["prices"].bfill()
events["reversed_clickout_step"] = (
    events.groupby(["action_type", "session_id"])["step"].rank("max", ascending=False).astype(int)
)
events["clickout_step"] = (
    events.groupby(["action_type", "session_id"])["step"].rank("max", ascending=True).astype(int)
)
events["clickout_max_step"] = events["clickout_step"] + events["reversed_clickout_step"] - 1
events["dt"] = events["timestamp"].apply(lambda x: str(arrow.get(x).date()))

num_of_validation_clickouts = len(events[(events["dt"] == "2018-11-06") & (events["action_type"] == "clickout item")]["session_id"].drop_duplicates())
num_of_clickouts = len(events[events["action_type"] == "clickout item"]["session_id"].drop_duplicates())
print(f"Validation ratio {round(num_of_validation_clickouts/num_of_clickouts, 2)}")

validation_events = events.loc[(events["dt"] == "2018-11-06") & (events["action_type"] == "clickout item")].copy(deep=True)

validation_events["reversed_user_clickout_step"] = (
    validation_events.groupby(["action_type", "user_id"])["step"].rank("max", ascending=False).astype(int)
)

last_clickouts_in_validation = validation_events[validation_events["reversed_user_clickout_step"] == 1][["user_id", "session_id", "step"]]
last_clickouts_in_validation["is_val"] = 1

events = pd.merge(events, last_clickouts_in_validation, on=["user_id", "session_id", "step"], how="left")


events["is_val"].fillna(0, inplace=True)
events["is_val"] = events["is_val"].astype(int)

events.sort_values(["session_id", "timestamp"]).to_csv(EVENTS_WRITE_PATH)
