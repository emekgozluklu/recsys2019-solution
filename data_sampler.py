import os
import pandas as pd

TAKE_N_SESSIONS = 50000
DATA_PATH = os.path.join("data")  # change it with your data folder
WRITE_PATH = os.path.join("data", "sampled")

try:
    os.mkdir(WRITE_PATH)
except FileExistsError:
    pass

train_filepath = os.path.join(DATA_PATH, "train.csv")
test_filepath = os.path.join(DATA_PATH, "test.csv")

print("Reading data...")
raw_train_data = pd.read_csv(train_filepath)
raw_train_data["session_valid"] = 1

all_session_ids = raw_train_data['session_id'].drop_duplicates()

print("Number of sessions: ", len(all_session_ids))

sampled_session_ids = all_session_ids[:TAKE_N_SESSIONS]  # pick the session id's that will be used
sampled_raw_train_data = raw_train_data.loc[raw_train_data['session_id'].isin(sampled_session_ids)]  # apply filtering

print("Labeling invalid sessions...")
invalid_sess = []

for i, sess in sampled_raw_train_data.groupby("session_id"):
    session_clickouts = sess[sess["action_type"] == "clickout item"]
    if len(session_clickouts) == 0:
        invalid_sess.append(i)
        continue

    for sess_id, co in session_clickouts.iterrows():
        if co["reference"] not in co["impressions"].split("|"):
            invalid_sess.append(i)
            break

sampled_raw_train_data.loc[sampled_raw_train_data["session_id"].isin(invalid_sess), "session_valid"] = 0

sampled_raw_train_data.to_csv(os.path.join(WRITE_PATH, "train_sample.csv"))  # write selected sessions to a csv
