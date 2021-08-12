import pandas as pd
import os
from csv import DictReader
from helpers import *
from tqdm import tqdm
from itertools import groupby
from collections import defaultdict
from constants import ITEM_ACTIONS

EVENTS_PATH = os.path.join("data", "events.csv")
SAVE_PATH = os.path.join("data", "session_features.csv")

DUMMY = -1000


class UserFeatures:

    def __init__(self, data_path, write_path="data/session_features.csv", num_of_sessions=50000, events_sorted=False):

        self.data = DictReader(open(data_path, encoding='utf-8'))
        self.sorted = events_sorted
        self.data_sorted = self.data if self.sorted else None
        self.num_of_sessions = num_of_sessions
        self.write_path = write_path

        self.user_index = 0
        self.user_id = -1
        self.user_clickouts_until_now = None
        self.user_item_actions_until_now = None
        self.user_all_clickouts = None
        self.user_all_item_action = None
        self.user_interacted_items = None
        self.user_interacted_item_prices = None
        self.user_clicked_items = None
        self.user_clicked_item_prices = None
        self.user_item_interaction_map = None

        self.last_price_diff_general = []
        self.avg_price_similarity = []
        self.user_start_ts = []
        self.sessions_of_user = []
        self.global_avg_price_rank = []
        self.viewed_items_user = []
        self.interacted_items_user = []
        self.viewed_items_avg_price = []
        self.interacted_items_avg_price = []
        self.viewed_items_avg_price_div = []
        self.interacted_items_avg_price_div = []
        self.item_clicked_before = []

        self.feature_updater_map = {
            "last_price_diff_general": self.update_last_price_diff_general,
            "avg_price_similarity": self.update_avg_price_similarity,
            "user_start_ts": self.update_user_start_ts,
            "sessions_of_user": self.update_sessions_of_user,
            "global_avg_price_rank": self.update_global_avg_price_rank,
            "viewed_items_user": self.update_viewed_items_user,
            "interacted_items_user": self.update_interacted_items_user,
            "viewed_items_avg_price": self.update_viewed_items_avg_price,
            "interacted_items_avg_price": self.update_interacted_items_avg_price,
            "viewed_items_avg_price_div": self.update_viewed_items_avg_price_div,
            "interacted_items_avg_price_div": self.update_interacted_items_avg_price_div,
            "item_clicked_before": self.update_item_clicked_before,
        }
        self.feature_array_map = {
            "last_price_diff_general": self.last_price_diff_general,
            "avg_price_similarity": self.avg_price_similarity,
            "user_start_ts": self.user_start_ts,
            "sessions_of_user": self.sessions_of_user,
            "global_avg_price_rank": self.global_avg_price_rank,
            "viewed_items_user": self.viewed_items_user,
            "interacted_items_user": self.interacted_items_user,
            "viewed_items_avg_price": self.viewed_items_avg_price,
            "interacted_items_avg_price": self.interacted_items_avg_price,
            "viewed_items_avg_price_div": self.viewed_items_avg_price_div,
            "interacted_items_avg_price_div": self.interacted_items_avg_price_div,
            "item_clicked_before": self.item_clicked_before
        }

        self.feature_names = list(self.feature_array_map.keys())

    def update_last_price_diff_general(self):
        pass

    def update_avg_price_similarity(self):
        pass

    def update_user_start_ts(self):
        pass

    def update_sessions_of_user(self):
        pass

    def update_global_avg_price_rank(self):
        pass

    def update_viewed_items_user(self):
        pass

    def update_interacted_items_user(self):
        pass

    def update_viewed_items_avg_price(self):
        pass

    def update_interacted_items_avg_price(self):
        pass

    def update_viewed_items_avg_price_div(self):
        pass

    def update_interacted_items_avg_price_div(self):
        pass

    def update_item_clicked_before(self):
        pass

    def invalid_session_handler(self):
        pass

    def run_updaters(self):
        pass

    def extract_features(self):
        pass

    def save_features(self):
        pass


if __name__ == "__main__":
    print("No action yet.")
    # ufe = UserFeatures("data/events_sorted.csv", events_sorted=True)
    # ufe.extract_features()
    # print(ufe.invalid_session_ids)
    # print("Saving...")
    # csfe.save_features()
