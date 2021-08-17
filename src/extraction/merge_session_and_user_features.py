import pandas as pd
from tqdm import tqdm
import os

COMMON_FEATURES = ['session_id', 'timestamp', 'platform', 'device', 'active_filters', 'hour', 'day_of_the_week',
                   'user_id', 'last_action_type', 'city', "step", "sessions_of_user", "viewed_items_avg_price",
                   "interacted_items_avg_price"]

SESSION_FILE = os.path.join("../../data", "session_features.csv")
USER_FILE = os.path.join("../../data", "user_features.csv")


def test_dataframe_consistency(df, cols):
    for i in range(100):
        row = df.loc[i]
        lens = set()
        for col in cols:
            lens.add(len(row[col]))
            if len(lens) != 1:
                raise Exception("Data inconsistency!")


def get_cols_to_be_split(merged):
    to_be_split = []

    for col in merged.columns:  # first general features are eliminated
        if str(merged.dtypes[col]) == "object" and col not in COMMON_FEATURES:
            to_be_split.append(col)

    return to_be_split


def generate_records(merged, to_be_split):
    records = []
    for row_id, row in tqdm(merged.iterrows()):
        for i in range(int(row["num_of_impressions"])):
            record = []
            for col_name, col in zip(merged.columns, row):
                if col_name in to_be_split:
                    try:
                        record.append(col[i])
                    except:
                        print(col_name)
                else:
                    record.append(col)
            records.append(tuple(record))
    return records


def reindex_column_to_end(columns, col):
    columns.remove(col)
    columns.append(col)
    return columns


def merge(session_features=None, user_features=None, save=True):
    if session_features is None:
        session_features = pd.read_csv(SESSION_FILE, index_col=0)
    if user_features is None:
        user_features = pd.read_csv(USER_FILE, index_col=0)

    session_features = session_features.dropna(thresh=2)

    # While extracting session features, clickouts to a different item which does not shown in the impressions
    # are considered as invalid. This causes a shape inconsistency with 2 dataframes. We want to drop those invalid
    # row, so we merge user features onto session features.

    merged_valid = session_features.merge(user_features, how="left", on=["session_id", "timestamp"])
    to_be_split = get_cols_to_be_split(merged_valid)

    for col in to_be_split:
        merged_valid[col] = list(merged_valid[col].str.split("|"))

    records = generate_records(merged_valid, to_be_split)
    records_df = pd.DataFrame(records, columns=merged_valid.columns)
    del records

    records_df["timestamp"] = records_df["timestamp"].apply(int)

    cols = reindex_column_to_end(list(records_df.columns), "clicked")

    records_df = records_df[cols]

    if save:
        records_df.to_csv("../../data/user_session_merged.csv")

    return records_df


if __name__ == "__main__":
    merge()
