import os
from csv import DictReader
import joblib
from tqdm import tqdm
from itertools import groupby
import time
import pandas as pd
import numpy as np
from helpers import normalize_float

EVENTS_PATH = os.path.join("data", "events_sorted.csv")

DUMMY = -1000


class ItemFeatures:

    def __init__(self, data_path, write_path="data/all_features.csv", num_of_sessions=50000, events_sorted=False):

        self.data = DictReader(open(data_path, encoding='utf-8'))
        self.sorted = events_sorted
        self.data_sorted = self.data if self.sorted else None
        self.num_of_sessions = num_of_sessions
        self.write_path = write_path

        self.item_to_poi_map_df = joblib.load(os.path.join("data", "item_to_poi_map.joblib"))
        self.item_sort_by_distance_stats_map = joblib.load(os.path.join("data", "item_sort_by_distance_stats.joblib"))
        self.item_sort_by_popularity_stats_map = joblib.load(
            os.path.join("data", "item_sort_by_popularity_stats.joblib"))
        self.item_sort_by_rating_stats_map = joblib.load(os.path.join("data", "item_sort_by_rating_stats.joblib"))
        self.item_prices_features_df = pd.read_csv(os.path.join("data", "item_prices.csv"))
        self.item_dense_features_df = pd.read_csv(os.path.join("data", "item_dense_features.csv"))
        self.item_price_pct_by_platform_map = joblib.load(os.path.join("data", "price_pct_by_platform.joblib"))
        self.item_price_pct_by_city_map = joblib.load(os.path.join("data", "price_pct_by_city.joblib"))

        self.item_data = None
        self.current_item_id = None
        self.current_item_data = None
        self.data_sorted_df = None

        self.current_item_num_of_appears = None
        self.current_item_click_events = None
        self.current_item_num_of_clicks = None
        self.current_item_clicked_ranks = None
        self.current_item_clicked_price_ranks = None
        self.item_impressions_until_now = None
        self.current_item_historical_mean_rank = None
        self.clicked = None
        self.price = None
        self.price_rank = None
        self.num_of_interactions = None
        self.clicks_until_now = None
        self.platform = None
        self.city = None
        self.current_item_ctr = None

        self.final_df = None

        self.historical_mean_rank = []  # added
        self.clickout_item_ctr = []  # added
        self.mean_rank_counter_mean = []  # session-item feature
        self.mean_rank_counter_min = []  # session-item feature
        self.clickout_counter_vs_interaction_counter_pure = []  # added
        self.item_clickouts_intersection = []
        self.price_pct_by_platform = []  # added
        self.price_pct_by_city = []  # added
        self.sort_by_rating_stats = []  # added
        self.sort_by_distance_stats = []  # added
        self.sort_by_popularity_stats = []  # added
        self.clickout_item_ctr_corr_by_platform = []

        self.feature_updater_map = {
            "historical_mean_rank": self.update_historical_mean_rank,
            "clickout_item_ctr": self.update_clickout_item_ctr,
            # "mean_rank_counter_mean": self.update_mean_rank_counter_mean,  # session-item feature
            # "mean_rank_counter_min": self.update_mean_rank_counter_min,  # session-item feature
            # "clickout_item_ctr_rank_weighted": self.update_clickout_item_ctr_rank_weighted,
            # "clickout_item_ctr_corr": self.update_clickout_item_ctr_corr,  # session-item feature
            "clickout_counter_vs_interaction_counter_pure": self.update_clickout_counter_vs_interaction_counter_pure,
            # "item_clickouts_intersection": self.update_item_clickouts_intersection,
            "price_pct_by_platform": self.update_price_pct_by_platform,
            "price_pct_by_city": self.update_price_pct_by_city,
            "sort_by_rating_stats": self.update_sort_by_rating_stats,
            "sort_by_distance_stats": self.update_sort_by_distance_stats,
            "sort_by_popularity_stats": self.update_sort_by_popularity_stats,
        }

        self.feature_array_map = {
            "historical_mean_rank": self.historical_mean_rank,
            "clickout_item_ctr": self.clickout_item_ctr,
            # "mean_rank_counter_mean": self.mean_rank_counter_mean,
            # "mean_rank_counter_min": self.mean_rank_counter_min,
            "clickout_counter_vs_interaction_counter_pure": self.clickout_counter_vs_interaction_counter_pure,
            # "item_clickouts_intersection": self.item_clickouts_intersection,
            "price_pct_by_platform": self.price_pct_by_platform,
            "price_pct_by_city": self.price_pct_by_city,
            "sort_by_rating_stats": self.sort_by_rating_stats,
            "sort_by_distance_stats": self.sort_by_distance_stats,
            "sort_by_popularity_stats": self.sort_by_popularity_stats,
        }

        self.feature_names = list(self.feature_array_map.keys())

    def update_historical_mean_rank(self):
        self.historical_mean_rank.append(self.current_item_historical_mean_rank)

    def update_clickout_item_ctr(self):
        self.clickout_item_ctr.append(self.current_item_ctr)

    def update_mean_rank_counter_mean(self):
        pass

    def update_mean_rank_counter_min(self):
        pass

    def update_clickout_item_ctr_rank_weighted(self):
        pass

    def update_clickout_item_ctr_corr(self):
        pass

    def update_clickout_counter_vs_interaction_counter_pure(self):
        if self.num_of_interactions > 0:
            self.clickout_counter_vs_interaction_counter_pure.append(self.clicks_until_now / self.num_of_interactions)
        else:
            self.clickout_counter_vs_interaction_counter_pure.append(0)

    def update_item_clickouts_intersection(self):
        pass

    def update_price_pct_by_platform(self):
        self.price_pct_by_platform.append(
            self.item_price_pct_by_platform_map[(self.platform, self.price)]
        )

    def update_price_pct_by_city(self):
        self.price_pct_by_city.append(
            self.item_price_pct_by_city_map[(self.city, self.price)]
        )

    def update_sort_by_rating_stats(self):
        try:
            stats = self.item_sort_by_rating_stats_map[self.current_item_id]
        except KeyError:
            stats = 0
        self.sort_by_rating_stats.append(stats)

    def update_sort_by_distance_stats(self):
        try:
            stats = self.item_sort_by_distance_stats_map[self.current_item_id]
        except KeyError:
            stats = 0
        self.sort_by_distance_stats.append(stats)

    def update_sort_by_popularity_stats(self):
        try:
            stats = self.item_sort_by_popularity_stats_map[self.current_item_id]
        except KeyError:
            stats = 0
        self.sort_by_popularity_stats.append(stats)

    def add_dense_features(self):
        self.item_dense_features_df["item_id"] = self.item_dense_features_df["item_id"].apply(int)
        self.data_sorted_df["item_id"] = self.data_sorted_df["item_id"].apply(int)
        self.data_sorted_df = self.data_sorted_df.merge(self.item_dense_features_df, on="item_id", how="left")

    def add_item_features(self):
        self.validate_data()
        for feat in self.feature_names:
            self.data_sorted_df[feat] = self.feature_array_map[feat]

    def validate_data(self):
        lengths = set()
        for feat in self.feature_names:
            lengths.add(len(self.feature_array_map[feat]))
            if len(lengths) != 1:
                raise Exception(f"A size inconsistency occured at {feat}")
        return 1

    def run_updaters(self):
        for updater in self.feature_updater_map.values():
            updater()

    def sort_data(self):
        if not self.sorted:
            print("not sorted, sorting...")
            t = time.time()
            self.data_sorted = sorted(self.data, key=lambda x: (int(x["item_id"]), int(x["timestamp"])))
            self.sorted = True
            print("sorted.", normalize_float(time.time() - t), " seconds.")
        else:
            print("already sorted.")

    def extract_features(self):
        print("extracting item features.")
        if not self.sorted:
            self.sort_data()

        for item_id, item_data in tqdm(groupby(self.data_sorted, lambda x: x["item_id"])):

            self.current_item_id = item_id
            self.current_item_data = list(item_data)

            self.current_item_num_of_appears = len(self.current_item_data)
            self.current_item_click_events = list(filter(lambda x: x["clicked"] == "1", self.current_item_data))
            self.current_item_num_of_clicks = len(self.current_item_click_events)
            self.current_item_clicked_ranks = [int(x["rank"]) for x in self.current_item_click_events]
            self.current_item_clicked_price_ranks = [int(x["price_rank"]) for x in self.current_item_click_events]
            self.current_item_historical_mean_rank = np.mean([int(x["rank"]) for x in self.current_item_data])
            self.item_impressions_until_now = []
            self.clicks_until_now = 0

            for action in self.current_item_data:
                self.item_impressions_until_now.append(action)

                self.clicked = int(action["clicked"])
                self.price = int(action["price"])
                self.price_rank = int(action["price_rank"])
                self.num_of_interactions = int(action["num_of_item_actions"])
                self.platform = action["platform"]
                self.city = action["city"]

                self.current_item_ctr = self.clicks_until_now / len(self.item_impressions_until_now)

                if self.clicked:
                    self.clicks_until_now += 1

                self.run_updaters()

        self.data_sorted_df = pd.DataFrame(self.data_sorted)
        self.add_item_features()
        self.add_dense_features()

    def save_features(self):
        self.data_sorted_df.to_csv("data/all_features.csv")

    def get_features(self):
        return self.data_sorted_df


if __name__ == "__main__":
    ife = ItemFeatures("data/user_session_merged.csv", events_sorted=True)
    ife.extract_features()
    ife.save_features()
