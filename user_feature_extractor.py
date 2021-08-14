import os
from csv import DictReader
from tqdm import tqdm
from itertools import groupby
from collections import defaultdict
from constants import ITEM_ACTIONS
import pandas as pd
import numpy as np
from helpers import normalize_float, dummy_function

EVENTS_PATH = os.path.join("data", "events.csv")

DUMMY = -1000


class UserFeatures:

    def __init__(self, data_path, write_path="data/user_features.csv", num_of_sessions=50000, events_sorted=False):

        self.data = DictReader(open(data_path, encoding='utf-8'))
        self.sorted = events_sorted
        self.data_sorted = self.data if self.sorted else None
        self.num_of_sessions = num_of_sessions
        self.write_path = write_path

        self.user_index = 0
        self.user_id = -1

        self.user_data = None

        self.current_impressions = None
        self.current_prices = None
        self.current_reference = None
        self.current_action = None

        self.user_actions_until_now = None
        self.user_last_clickouts = None
        self.user_all_clickouts = None
        self.user_interacted_items = None
        self.user_interacted_item_prices = None
        self.user_clicked_items = None
        self.user_clicked_item_prices = None
        self.user_item_interaction_map = None
        self.user_item_actions_dist = None
        self.user_all_item_actions = None

        self.user_sessions = None

        item_prices_df = pd.read_csv(os.path.join("data", "item_prices.csv"))
        mean_item_prices = item_prices_df.groupby("item_id")["price"].mean().to_dict()
        self.mean_item_prices = defaultdict(int, mean_item_prices)

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
        self.clickout_user_id = []
        self.clickout_session_id = []
        self.clickout_timestamp = []

        self.feature_updater_map = {
            "last_price_diff_general": self.update_last_price_diff_general,
            "avg_price_similarity": self.update_avg_price_similarity,
            "user_start_ts": self.update_user_start_ts,
            "sessions_of_user": self.update_sessions_of_user,
            # "global_avg_price_rank": self.update_global_avg_price_rank,
            "viewed_items_user": self.update_viewed_items_user,
            "interacted_items_user": self.update_interacted_items_user,
            "viewed_items_avg_price": self.update_viewed_items_avg_price,
            "interacted_items_avg_price": self.update_interacted_items_avg_price,
            "viewed_items_avg_price_div": self.update_viewed_items_avg_price_div,
            "interacted_items_avg_price_div": self.update_interacted_items_avg_price_div,
            "item_clicked_before": self.update_item_clicked_before,
            "clickout_user_id": self.update_action_identifiers,
            "clickout_session_id": dummy_function,
            "clickout_timestamp": dummy_function,
        }
        self.feature_array_map = {
            "last_price_diff_general": self.last_price_diff_general,
            "avg_price_similarity": self.avg_price_similarity,
            "user_start_ts": self.user_start_ts,
            "sessions_of_user": self.sessions_of_user,
            # "global_avg_price_rank": self.global_avg_price_rank,
            "viewed_items_user": self.viewed_items_user,
            "interacted_items_user": self.interacted_items_user,
            "viewed_items_avg_price": self.viewed_items_avg_price,
            "interacted_items_avg_price": self.interacted_items_avg_price,
            "viewed_items_avg_price_div": self.viewed_items_avg_price_div,
            "interacted_items_avg_price_div": self.interacted_items_avg_price_div,
            "item_clicked_before": self.item_clicked_before,
            "clickout_user_id": self.clickout_user_id,
            "clickout_session_id": self.clickout_session_id,
            "clickout_timestamp": self.clickout_timestamp,
        }

        self.feature_names = list(self.feature_array_map.keys())

    def update_last_price_diff_general(self):
        if len(self.user_clicked_item_prices) > 1:
            last_co_price = int(self.user_clicked_item_prices[-1])
            self.last_price_diff_general.append("|".join([str(pri - last_co_price) for pri in self.current_prices]))
        else:
            self.last_price_diff_general.append(DUMMY)

    def update_avg_price_similarity(self):
        if len(self.user_clicked_item_prices) > 0:
            avg_price = np.mean(self.user_clicked_item_prices)
            self.avg_price_similarity.append(
                "|".join(str(normalize_float(pri-avg_price)) for pri in self.current_prices))
        else:
            self.avg_price_similarity.append(DUMMY)

    def update_user_start_ts(self):
        self.user_start_ts.append(
            int(self.current_action["timestamp"]) - int(self.user_actions_until_now[0]["timestamp"])
        )

    def update_sessions_of_user(self):
        sessions_until_now = set([x["session_id"] for x in self.user_actions_until_now])
        self.sessions_of_user.append(len(sessions_until_now))

    def update_global_avg_price_rank(self):
        pass

    def update_viewed_items_user(self):
        self.viewed_items_user.append("|".join(["1" if self.current_reference in self.user_clicked_items else "0"]))

    def update_interacted_items_user(self):
        self.interacted_items_user.append("|".join(["1" if self.current_reference in self.user_interacted_items else "0"]))

    def update_viewed_items_avg_price(self):
        if len(self.user_clicked_item_prices) > 0:
            self.viewed_items_avg_price.append(normalize_float(np.mean(self.user_clicked_item_prices)))
        else:
            self.viewed_items_avg_price.append(DUMMY)

    def update_interacted_items_avg_price(self):
        if len(self.user_interacted_item_prices) > 0:
            self.interacted_items_avg_price.append(np.mean(self.user_interacted_item_prices))
        else:
            self.interacted_items_avg_price.append(DUMMY)

    def update_viewed_items_avg_price_div(self):
        if len(self.user_clicked_item_prices) > 0:
            avg = np.mean(self.user_clicked_item_prices)
            self.viewed_items_avg_price_div.append(
                "|".join([str(normalize_float(pri/avg)) for pri in self.current_prices])
            )
        else:
            self.viewed_items_avg_price_div.append(DUMMY)

    def update_interacted_items_avg_price_div(self):
        if len(self.user_interacted_item_prices) > 0:
            avg = np.mean(self.user_interacted_item_prices)
            self.interacted_items_avg_price_div.append(
                "|".join([str(normalize_float(pri/avg)) for pri in self.current_prices])
            )
        else:
            self.interacted_items_avg_price_div.append(DUMMY)

    def update_item_clicked_before(self):
        self.item_clicked_before.append("|".join([str(int(imp_id in self.user_interacted_items)) for imp_id in self.current_impressions]))

    def invalid_session_handler(self):
        self.last_price_diff_general.append(None)

    def update_action_identifiers(self):
        self.clickout_user_id.append(self.current_action["user_id"])
        self.clickout_session_id.append(self.current_action["session_id"])
        self.clickout_timestamp.append(self.current_action["timestamp"])

    def run_updaters(self):
        for updater in self.feature_updater_map.values():
            updater()

    def filter_by_session_id(self, session_id):
        return list(
            filter(lambda x: x["session_id"] == session_id, self.user_data)
        )

    def filter_by_action_type(self, action_type):
        return list(
            filter(lambda x: x["action_type"] == action_type, self.user_data)
        )

    def get_last_clickouts_of_user(self):
        last_clickouts = dict()
        for sid in self.user_sessions:
            sess_clickouts = filter(
                lambda x: x["session_id"] == sid and x["action_type"] == "clickout item",
                self.user_data
            )
            for co in sess_clickouts:
                last_clickouts[co["session_id"]] = co
        return list(last_clickouts.values())

    def extract_features(self):
        print("extracting user features.")
        self.data_sorted = sorted(self.data, key=lambda x: (x["user_id"], ["timestamp"]))
        self.sorted = True

        for user_id, user_actions in tqdm(groupby(self.data_sorted, lambda x: x["user_id"])):

            # predefined variables
            self.user_id = user_id
            self.user_data = list(user_actions)
            self.user_all_clickouts = self.filter_by_action_type("clickout item")
            self.user_all_item_actions = list(
                filter(
                    lambda x: x["action_type"] in ITEM_ACTIONS and x["reference"] != "unknown",
                    self.user_data
                )
            )

            self.user_sessions = set([x["session_id"] for x in self.user_data])
            self.user_last_clickouts = self.get_last_clickouts_of_user()

            # accumulated features
            self.user_clicked_items = []
            self.user_interacted_items = []
            self.user_interacted_item_prices = []
            self.user_item_interaction_map = defaultdict(int)
            self.user_item_actions_dist = defaultdict(int)
            self.user_actions_until_now = []
            self.user_clicked_item_prices = []

            for action in self.user_data:

                self.user_actions_until_now.append(action)

                action_type = action["action_type"]
                reference = action["reference"]
                prices = action["prices"].split("|")
                impressions = action["impressions"].split("|")

                if action in self.user_last_clickouts:
                    self.current_impressions = list(map(int, impressions))
                    self.current_prices = list(map(int, prices))
                    self.current_reference = int(reference)
                    self.current_action = action

                    # it is important to run updaters before accumulated feature get updated
                    # we do not want to use current reference yet.
                    self.run_updaters()

                # updaters ran, we can update accumulators
                if action_type in ITEM_ACTIONS and reference != "unknown":
                    self.user_interacted_items.append(int(reference))
                    self.user_interacted_item_prices.append(
                        int(prices[impressions.index(reference)])
                        if reference in impressions
                        else self.mean_item_prices[int(reference)]
                    )
                    self.user_item_interaction_map[(user_id, reference)] += 1
                    self.user_item_actions_dist[reference] += 1

                    if action_type == "clickout item":
                        self.user_clicked_items.append(int(reference))
                        self.user_clicked_item_prices.append(
                            int(prices[impressions.index(reference)])
                            if reference in impressions
                            else self.mean_item_prices[int(reference)]
                        )

    def save_features(self):
        as_df = pd.DataFrame(columns=self.feature_names)
        for feat, values in self.feature_array_map.items():
            print(feat, len(values))
            as_df[feat] = values
        as_df.to_csv(self.write_path)


if __name__ == "__main__":
    print("No action yet.")
    ufe = UserFeatures("data/events_sorted.csv", events_sorted=True)
    ufe.extract_features()
    ufe.save_features()
