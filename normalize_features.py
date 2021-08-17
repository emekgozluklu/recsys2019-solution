import os
import joblib
import pandas as pd
from helpers import normalize_feature_name
from src.constants import ITEM_ACTIONS
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessing:

    def __init__(self, features_file="data/all_features.csv"):

        self.features = pd.read_csv(features_file)
        self.table_cols = list(self.features.columns)

        self.minmax_scaler = MinMaxScaler()
        self.standart_scaler = StandardScaler()

        self.drop = ["session_id", "city", "item_id", "user_id", "item_index", "timestamp"]

        self.one_hot = ["platform", "device", "day_of_the_week", "rating", "stars", "hotel_category"]

        self.custom_one_hot = ["step", "last_action_type", "sessions_of_user", "hour"]

        self.pivot = ["active_filters"]

        self.keep = [
            "cheapest", "most_expensive", "price_rel_to_imp_min", "price_rel_to_imp_max", "rank", "price_rank",
            "price_rank_among_above", "price_vs_mean_price", "top_of_impression", "clickout_prob_time_position_offset",
            "user_viewed_item", "item_clicked_before", "user_interacted_item", "viewed_items_avg_price_div",
            "interacted_items_avg_price_div", "clickout_item_ctr", "price_pct_by_platform", "price_pct_by_city",
            "sort_by_rating_stats", "sort_by_distance_stats", "sort_by_popularity_stats", "free_wifi_(combined)",
            "swimming_pool_(combined_filter)", "car_park", "serviced_apartment", "air_conditioning",
            "spa_(wellness_facility)", "pet_friendly", "all_inclusive_(upon_inquiry)",
        ]

        self.log_scale_minmax = ["time_since_last_item_action", "user_start_ts"]

        self.minmax = [
            "price", "impressions_max_price", "impressions_min_price", "impressions_price_range", "num_of_impressions",
            "session_start_ts", "clickout_item_item_last_timestamp", "num_of_item_actions",
            "last_interaction_absolute_position", "actions_since_last_item_action", "same_impression_in_session",
            "price_above", "viewed_items_avg_price", "interacted_items_avg_price",
            "interacted_and_viewed_items_price_diff", "historical_mean_rank",
            "clickout_counter_vs_interaction_counter_pure",
        ]

        self.normalize_on_zero = [
            "previous_click_price_diff_session", "price_rank_diff_last_interaction",
            "last_interaction_relative_position", "avg_price_similarity", "last_price_diff_general",
            "viewed_items_avg_price_diff", "interacted_items_avg_price_diff",
        ]

        self.target = ["clicked"]

        self.all_features = self.one_hot + self.drop + self.pivot + self.keep + self.log_scale_minmax + self.minmax +\
            self.normalize_on_zero + self.target + self.custom_one_hot

        for col in self.table_cols:
            if col not in self.all_features:
                self.drop.append(col)

    def apply_drop(self):
        self.features = self.features.drop(columns=self.drop)

    def apply_pivot(self):

        def split_if_not_none(text):
            if type(text) is float:
                return None
            return text.split("|")

        def contains(x, feature):
            if type(x) is float:
                return 0
            if feature in x:
                return 1
            return 0

        for piv in self.pivot:
            feats = set()

            items = self.features[piv].apply(split_if_not_none).dropna()
            indexes = items.index

            items_as_list = list(items)
            for item in items_as_list:
                for feat in item:
                    feats.add(normalize_feature_name(feat))

            feats = list(feats)
            self.features[feats] = 0

            for feat in feats:
                self.features.loc[indexes, feat] = self.features.loc[indexes, piv].apply(contains, args=(feat,))

            self.features = self.features.drop(columns=[piv])

    def apply_one_hot(self):
        for feat in self.one_hot:
            values = pd.get_dummies(self.features[feat], prefix=feat)
            self.features = pd.concat([self.features, values], axis=1)
            self.features.drop(columns=[feat], inplace=True)

    def apply_custom_one_hot(self):
        step_vals = self.features["step"]
        cols = {
            "1": step_vals == 1,
            "2": step_vals == 2,
            "3": step_vals == 3,
            "4": step_vals == 4,
            "5": step_vals == 5,
            "6_10": (step_vals >= 5) & (step_vals <= 10),
            "11_25": (step_vals >= 11) & (step_vals <= 25),
            "26_more": step_vals >= 25,
        }
        for name, value in cols.items():
            self.features["step_" + name] = value.astype(int)

        last_action_type = self.features["last_action_type"]
        cols = {
            "not_item_action": ~last_action_type.isin(ITEM_ACTIONS),
            "item_action": last_action_type.isin(ITEM_ACTIONS),
        }
        for name, value in cols.items():
            self.features["last_action_type_" + name] = value.astype(int)

        sessions_of_user = self.features["sessions_of_user"]
        cols = {
            "1": sessions_of_user == 1,
            "2": sessions_of_user == 2,
            "3_more": sessions_of_user >= 3,
        }

        for name, value in cols.items():
            self.features["sessions_of_user_" + name] = value.astype(int)

        hour = self.features["hour"]
        cols = {
            "0_5": (hour <= 5),
            "6_11": (hour >= 6) & (hour <= 11),
            "12_18": (hour >= 12) & (hour <= 18),
            "19_24": (hour >= 19) & (hour <= 24),
        }

        for name, value in cols.items():
            self.features["nour_" + name] = value.astype(int)

    def apply_minmax(self):
        for feat in self.log_scale_minmax:
            self.features.loc[self.features[feat] > 1, feat] = np.ceil(np.log(self.features.loc[self.features[feat] > 1, feat]))
            if feat not in self.minmax:
                self.minmax.append(feat)

        self.minmax_scaler.fit(self.features[self.minmax])
        self.features[self.minmax] = self.minmax_scaler.transform(self.features[self.minmax])

        if "scalers" not in os.listdir():
            os.mkdir("scalers")

        joblib.dump(self.minmax_scaler, "scalers/minmax_scaler.joblib")

    def apply_normalize_on_zero(self):
        self.standart_scaler.fit(self.features[self.normalize_on_zero])
        self.features[self.normalize_on_zero] = self.standart_scaler.transform(self.features[self.normalize_on_zero])

        if "scalers" not in os.listdir():
            os.mkdir("scalers")

        joblib.dump(self.minmax_scaler, "scalers/standart_scaler.joblib")

    def run(self, verbose=0):
        print("Nomalization started.")
        self.apply_drop()
        self.apply_pivot()
        self.apply_one_hot()
        self.apply_custom_one_hot()
        self.apply_minmax()
        self.apply_normalize_on_zero()

    def save(self, path="data/normalized_features.csv"):
        self.features.to_csv(path)


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.run(verbose=1)
    dp.save()
