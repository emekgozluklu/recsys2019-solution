# This file includes software licensed under the Apache License 2.0, modified by LogicAI Co.
# https://github.com/logicai-io/recsys2019/blob/master/src/recsys/data_prep/extract_city_prices_percentiles.py

from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm


def run():
    df = pd.read_csv("../../data/events_sorted.csv")

    for criterion in ["Sort By Rating", "Sort By Distance", "Sort By Popularity"]:
        name = criterion.lower().replace(" ", "_")

        df_sort_by_rating = df[(df["current_filters"].str.find(criterion) >= 0) & (df["action_type"] == "clickout item")].copy()
        df_sort_by_rating["impressions_parsed"] = (
            df_sort_by_rating["impressions"].str.split("|").map(lambda x: list(map(int, x)))
        )

        ordered_pairs = set()
        all_items = set()
        for impressions in tqdm(df_sort_by_rating["impressions_parsed"]):
            for idx_a, item_a in enumerate(impressions):
                for idx_b, item_b in enumerate(impressions[(idx_a + 1):]):
                    ordered_pairs.add((item_a, item_b))
                    all_items.add(item_a)
                    all_items.add(item_b)

        good_item = defaultdict(int)
        bad_item = defaultdict(int)

        for item_a, item_b in ordered_pairs:
            good_item[item_a] += 1
            bad_item[item_b] += 1

        stats = defaultdict(float)
        for item_id in all_items:
            stats[item_id] = good_item[item_id] / (good_item[item_id] + bad_item[item_id])

        joblib.dump(stats, f"../../data/item_{name}_stats.joblib")
        print(f"../../data/item_{name}_stats.joblib saved!")


if __name__ == "__main__":
    run()
