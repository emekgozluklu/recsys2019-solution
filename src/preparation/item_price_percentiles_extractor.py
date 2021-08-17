# This file includes software licensed under the Apache License 2.0, modified by LogicAI Co.
# https://github.com/logicai-io/recsys2019/blob/master/src/recsys/data_prep/extract_city_prices_percentiles.py

import joblib
import pandas as pd
from tqdm import tqdm


def run():
    df = pd.read_csv("../../data/events_sorted.csv")
    clickouts = df[df["action_type"] == "clickout item"]

    obs = set()

    for city, platform, impressions_list, prices_list in tqdm(
        zip(clickouts["city"], clickouts["platform"], clickouts["impressions"], clickouts["prices"])
    ):
        items_ids = list(map(int, impressions_list.split("|")))
        prices = list(map(int, prices_list.split("|")))
        for item_id, price in zip(items_ids, prices):
            obs.add((city, platform, item_id, price))

    df_prices = pd.DataFrame.from_records(list(obs), columns=["city", "platform", "item_id", "price"])

    for key in ["city", "platform"]:
        df_prices[f"price_pct"] = df_prices.groupby(key)["price"].rank(pct=True)
        df_only_prices = df_prices.drop_duplicates([key, "price"]).drop("item_id", axis=1)
        price_pct = dict(zip(zip(df_only_prices[key], df_only_prices["price"]), df_only_prices["price_pct"]))
        joblib.dump(price_pct, f"../../data/price_pct_by_{key}.joblib", compress=3)


if __name__ == "__main__":
    run()
