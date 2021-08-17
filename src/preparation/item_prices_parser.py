from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
import numpy as np


def run():
    events = pd.read_csv(os.path.join("../../data", "events_sorted.csv"), index_col=0)
    clickout_events = events[events["action_type"] == "clickout item"].to_numpy()

    cols = list(events.columns)
    impression_index = cols.index("impressions")
    prices_index = cols.index("prices")

    impression_occurences = defaultdict(list)
    records = list()

    for co in tqdm(clickout_events):
        item_ids = co[impression_index].split("|")
        prices = co[prices_index].split("|")
        for item, price in zip(item_ids, prices):
            impression_occurences[item].append(int(price))
            records.append((item, int(price)))

    item_prices = pd.DataFrame.from_records(records, columns=["item_id", "price"]).drop_duplicates()
    item_prices.sort_values(["item_id", "price"], inplace=True)

    item_prices["ascending_price_rank"] = item_prices.groupby("item_id")["price"].rank("max", ascending=True)
    item_prices["descending_price_rank"] = item_prices.groupby("item_id")["price"].rank("max", ascending=False)
    item_prices["ascending_price_rank_pct"] = item_prices.groupby("item_id")["price"].rank("max", pct=True, ascending=True)

    aggregate_feature_recs = list()

    for item, prices in tqdm(impression_occurences.items()):
        aggregate_feature_recs.append((item, min(prices), max(prices), len(prices), np.mean(prices), np.std(prices)))

    aggreagate_features = pd.DataFrame.from_records(aggregate_feature_recs, columns=[
        "item_id", "min_price", "max_price", "price_count", "mean_price", "price_deviation"
    ])

    item_prices = pd.merge(item_prices, aggreagate_features, on="item_id")
    item_prices["price_relative_to_min"] = item_prices["price"] / item_prices["min_price"]
    item_prices["price_range"] = item_prices["max_price"] - item_prices["min_price"]
    item_prices["price_range_div"] = item_prices["max_price"] / item_prices["min_price"]

    item_prices.to_csv(os.path.join("../../data", "item_prices.csv"), index=False)


if __name__ == "__main__":
    run()
