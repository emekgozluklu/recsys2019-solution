import pandas as pd
from tqdm import tqdm

session_features = pd.read_csv("data/session_features.csv", index_col=0)
user_features = pd.read_csv("data/user_features.csv", index_col=0)

session_features = session_features.dropna(thresh=2)  # 2 NaN values accepted, others not valid

# While extracting session features, clickouts to a different item which does not shown in the impressions
# are considered as invalid. This causes a shape inconsistency with 2 dataframes. We want to drop those invalid
# row, so we merge user features onto session features.

merged_valid = session_features.merge(user_features, how="left", on=["session_id", "timestamp"])

to_be_splitted = []

BASIC_FEATURES = [ 'session_id', 'timestamp', 'platform', 'device', 'active_filters', 'hour', 'day_of_the_week',
                   'user_id', 'last_action_type' ]


for col in merged_valid.columns[6:]:  # first general features are eliminated
    if str(merged_valid.dtypes[col]) == "object" and col not in BASIC_FEATURES:
        to_be_splitted.append(col)

for col in to_be_splitted:
    print(col)
    merged_valid[col] = list(merged_valid[col].str.split("|"))

# test the data
for i in range(100):
    row = merged_valid.loc[i]
    lens = set()
    for col in to_be_splitted:
        lens.add(len(row[col]))
        if len(lens) != 1:
            raise Exception("Data inconsistency!")

records = []

for row_id, row in tqdm(merged_valid.iterrows()):
    for i in range(int(row["num_of_impressions"])):
        record = []
        for col_name, col in zip(merged_valid.columns, row):
            if col_name in to_be_splitted:
                record.append(col[i])
            else:
                record.append(col)
        records.append(tuple(record))

records_df = pd.DataFrame(records, columns=merged_valid.columns)
del records

records_df.to_csv("data/preprocessing_input.csv")
