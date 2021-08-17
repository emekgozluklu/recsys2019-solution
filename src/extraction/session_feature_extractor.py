import pandas as pd
import os
from csv import DictReader
from src.helpers import *
from tqdm import tqdm
from itertools import groupby
from collections import defaultdict
from src.constants import ITEM_ACTIONS
import joblib
import arrow

SAVE_PATH = os.path.join("../../data", "session_features.csv")
CLICK_PROBS_PATH = os.path.join("../../data", "click_probs_by_index.joblib")

DUMMY = -1000


class SessionFeatures:

    def __init__(self, data_path="../../data/events_sorted.csv", write_path="../../data/session_features.csv", num_of_sessions=50000, events_sorted=False):

        self.data = DictReader(open(data_path, encoding='utf-8'))
        self.sorted = events_sorted
        self.data_sorted = self.data if self.sorted else None
        self.num_of_sessions = num_of_sessions
        self.write_path = write_path
        self.clickout_probs = joblib.load(CLICK_PROBS_PATH)

        self.item_prices = DictReader(open("../../data/item_prices.csv"))

        self.current_session_index = 0
        self.current_session_id = -1
        self.current_session = None
        self.current_session_valid = True
        self.current_session_clickouts = None
        self.current_session_item_actions = None
        self.current_impressions = None
        self.current_prices = None
        self.datetime = None
        self.timestamp = None
        self.reference = None
        self.current_city = None

        self.session_id = []  # added
        self.session_start_ts = []  # added
        self.price_vs_mean_price = []
        self.clickout_item_item_last_timestamp = []  # added
        self.rank = []  # added
        self.previous_click_price_diff_session = []  # added
        self.time_since_last_image_interaction = []  # added
        self.top_of_impression = []  # added
        self.impression_star_ranking = []
        self.price_rank = []  # added
        self.price_rank_among_above = []  # added
        self.num_of_item_actions = []  # added
        self.price_rank_diff_last_interaction = []  # added
        self.price_rank_same_star_rating = []
        self.last_interaction_relative_position = []  # added
        self.last_interaction_absolute_position = []  # added
        self.actions_since_last_item_action = []  # added, eliminate ourliers
        self.time_since_last_item_action = []  # added
        self.same_star_rating_items = []
        self.same_user_rating_items = []
        self.same_impressions_in_session = []  # added
        self.step = []  # added
        self.last_action_type = []  # added
        self.price_above = []  # added
        self.price_diff_from_estimated_budget = []
        self.clickout_prob_time_position_offset = []  # added
        self.platform = []  # added
        self.device = []  # added
        self.active_filters = []  # added
        self.hour = []  # added
        self.day_of_the_week = []  # added
        self.cheapest = []  # added
        self.most_expensive = []  # added
        self.timestamps = []
        self.impressions_max_price = []
        self.impressions_min_price = []
        self.impressions_price_range = []
        self.num_of_impressions = []
        self.price_rel_to_imp_min = []
        self.price_rel_to_imp_max = []
        self.price_rel_to_hist_min = []
        self.price_rel_to_hist_max = []
        self.historical_mean_rank = []
        self.rank_among_historical_mean_ranks = []
        self.price = []
        self.item_id = []
        self.city = []

        self.clicked = []

        self.feature_updater_map = {
            "session_id": self.update_session_id,
            "session_start_ts": self.update_session_start_ts,
            "price_vs_mean_price": self.update_price_vs_mean_price,
            "clickout_item_item_last_timestamp": self.update_clickout_item_item_last_timestamp,
            "rank": self.update_rank,
            "previous_click_price_diff_session": self.update_previous_click_price_diff_session,
            "top_of_impression": self.update_top_of_impression,  # 7595 it/s
            "price_rank": self.update_price_rank,  # 7047 it/s
            "price_rank_among_above": self.update_price_rank_among_above,  # 6640 it/s
            "num_of_item_actions": self.update_num_of_item_actions,
            "price_rank_diff_last_interaction": self.update_price_rank_diff_last_interaction,  # 5582 it/s
            "last_interaction_relative_position": dummy_function,
            "last_interaction_absolute_position": dummy_function,
            "actions_since_last_item_action": dummy_function,
            "time_since_last_item_action": dummy_function,
            "same_impression_in_session": self.update_same_impressions_in_session,
            "step": self.update_step,
            "last_action_type": self.update_last_action_type,
            "price_above": self.update_price_above,
            "clickout_prob_time_position_offset": self.update_clickout_prob_time_position_offset,
            "platform": self.update_platform,
            "device": self.update_device,
            "active_filters": self.update_active_filters,
            "hour": self.update_hour,
            "day_of_the_week": self.update_day_of_the_week,
            "cheapest": self.update_cheapest,
            "most_expensive": self.update_most_expensive,
            "timestamps": self.update_timestamps,
            "impressions_max_price": self.update_impressions_max_price,
            "impressions_min_price": self.update_impressions_min_price,
            "impressions_price_range": self.update_impressions_price_range,
            "num_of_impressions": self.update_num_of_impressions,
            "price_rel_to_imp_min": self.update_price_rel_to_imp_min,
            "price_rel_to_imp_max": self.update_price_rel_to_imp_max,
            "price": self.update_price,
            "item_id": self.update_item_id,
            "clicked": self.update_clicked,
            "city": self.update_city,
            # "price_rel_to_hist_min": self.price_rel_to_hist_min,
            # "price_rel_to_hist_max": self.price_rel_to_hist_max,
            # "historical_mean_rank": self.historical_mean_rank,
            # "rank_among_historical_mean_ranks": self.rank_among_historical_mean_ranks,
        }

        self.feature_array_map = {
            "session_id": self.session_id,
            "timestamps": self.timestamps,
            "platform": self.platform,
            "device": self.device,
            "active_filters": self.active_filters,
            "hour": self.hour,
            "day_of_the_week": self.day_of_the_week,
            "price": self.price,
            "city": self.city,
            "item_id": self.item_id,
            "cheapest": self.cheapest,
            "most_expensive": self.most_expensive,
            "impressions_max_price": self.impressions_max_price,
            "impressions_min_price": self.impressions_min_price,
            "impressions_price_range": self.impressions_price_range,
            "num_of_impressions": self.num_of_impressions,
            "price_rel_to_imp_min": self.price_rel_to_imp_min,
            "price_rel_to_imp_max": self.price_rel_to_imp_max,
            "rank": self.rank,
            "price_rank": self.price_rank,
            "price_rank_among_above": self.price_rank_among_above,
            "session_start_ts": self.session_start_ts,
            "price_vs_mean_price": self.price_vs_mean_price,
            "clickout_item_item_last_timestamp": self.clickout_item_item_last_timestamp,
            "previous_click_price_diff_session": self.previous_click_price_diff_session,
            "top_of_impression": self.top_of_impression,
            "num_of_item_actions": self.num_of_item_actions,
            "price_rank_diff_last_interaction": self.price_rank_diff_last_interaction,  # 5582 it/s
            "last_interaction_relative_position": self.last_interaction_relative_position,
            "last_interaction_absolute_position": self.last_interaction_absolute_position,
            "actions_since_last_item_action": self.actions_since_last_item_action,
            "time_since_last_item_action": self.time_since_last_item_action,
            "same_impression_in_session": self.same_impressions_in_session,
            "step": self.step,
            "last_action_type": self.last_action_type,
            "price_above": self.price_above,
            "clickout_prob_time_position_offset": self.clickout_prob_time_position_offset,
            # "price_rel_to_hist_min": self.update_price_rel_to_hist_min,
            # "price_rel_to_hist_max": self.update_price_rel_to_hist_max,
            # "historical_mean_rank": self.update_historical_mean_rank,
            # "rank_among_historical_mean_ranks": self.update_rank_among_historical_mean_ranks,

            "clicked": self.clicked,
        }

        self.feature_names = list(self.feature_array_map.keys())

        self.invalid_session_ids = []

    def update_session_id(self):
        self.session_id.append(self.current_session_id)

    def update_session_start_ts(self):
        start_ts = int(self.current_session[0]["timestamp"])
        end_ts = int(self.current_session[-1]["timestamp"])
        self.session_start_ts.append(end_ts - start_ts)

    def update_price_vs_mean_price(self):
        mean_price = sum(self.current_prices) / len(self.current_prices)
        self.price_vs_mean_price.append("|".join([str(round(pri/mean_price, 3)) for pri in self.current_prices]))

    def update_clickout_item_item_last_timestamp(self):
        if len(self.current_session_clickouts) > 2:
            self.clickout_item_item_last_timestamp.append(
                int(self.current_session_clickouts[-1]["timestamp"]) -
                int(self.current_session_clickouts[-2]["timestamp"])
            )
        else:
            self.clickout_item_item_last_timestamp.append(0)

    def update_rank(self):
        self.rank.append("|".join([str(j+1) for j in range(len(self.current_impressions))]))

    def update_previous_click_price_diff_session(self):
        if len(self.current_session_clickouts) > 2:
            prev_item = self.current_session_clickouts[-2]["reference"]
            prev_pri = get_hotel_price_by_id(prev_item,
                                             self.current_session_clickouts[-2]["impressions"],
                                             self.current_session_clickouts[-2]["prices"])
            price_diffs = []
            for pri in self.current_prices:
                price_diffs.append(pri - prev_pri)

            self.previous_click_price_diff_session.append("|".join(map(str, price_diffs)))
        else:
            self.previous_click_price_diff_session.append("|".join(["0"]*len(self.current_prices)))

    def update_time_since_last_image_interaction(self):
        image_interactions = list(filter(lambda x: x["action_type"] == "interaction item image", self.current_session))
        if len(image_interactions) > 0:
            last_image_interaction_time = int(image_interactions[-1]["timestamp"])
            current_clickout_time = int(self.current_session_clickouts[-1]["timestamp"])
            self.time_since_last_image_interaction.append(current_clickout_time - last_image_interaction_time)
        else:
            self.time_since_last_image_interaction.append(0)

    def update_top_of_impression(self):
        self.top_of_impression.append("1|" + "|".join(["0"] * (len(self.current_impressions) - 1)))

    def update_price_rank(self):
        ranks = get_price_ranks_from_prices_list(self.current_prices)
        self.price_rank.append("|".join(map(str, ranks)))

    def update_price_rank_among_above(self):
        ranks = []
        prices = []
        for pri in self.current_prices:
            prices.append(pri)
            prices.sort()
            ranks.append(prices.index(pri)+1)
        self.price_rank_among_above.append("|".join(map(str, ranks)))

    def update_num_of_item_actions(self):
        session_item_actions = defaultdict(int)
        for action in self.current_session[:-1]:
            if action["action_type"] in ITEM_ACTIONS:
                if action["reference"] == "unknown":
                    continue
                session_item_actions[int(action["reference"])] += 1
        self.num_of_item_actions.append("|".join([str(session_item_actions[x]) for x in self.current_impressions]))

    def update_price_rank_diff_last_interaction(self):
        if len(self.current_session_item_actions) < 2 or int(self.current_session_item_actions[-2]["reference"]) not in self.current_impressions:
            self.price_rank_diff_last_interaction.append("|".join(["0"]*len(self.current_impressions)))
            self.last_interaction_relative_position.append("|".join(["0"]*len(self.current_impressions)))
            self.last_interaction_absolute_position.append(0)
            self.actions_since_last_item_action.append(len(self.current_session) - 1)
            self.time_since_last_item_action.append(0)
            return

        prev_item = int(self.current_session_item_actions[-2]["reference"])
        prev_item_pos = self.current_impressions.index(prev_item)
        price_ranks = get_price_ranks_from_prices_list(self.current_prices)
        prev_item_rank = price_ranks[prev_item_pos]

        last_co_step = int(self.current_session_clickouts[-1]["step"])
        last_co_ts = int(self.current_session_clickouts[-1]["timestamp"])

        item_actions_before_co = list(filter(lambda x: int(x["step"]) < last_co_step, self.current_session_item_actions))

        self.price_rank_diff_last_interaction.append("|".join([str(r - prev_item_rank) for r in price_ranks]))
        self.last_interaction_relative_position.append("|".join([str(pos - prev_item_pos) for pos in range(len(self.current_impressions))]))
        self.last_interaction_absolute_position.append(prev_item_pos)
        if item_actions_before_co:
            self.actions_since_last_item_action.append(last_co_step - int(item_actions_before_co[-1]["step"]) - 1)
            self.time_since_last_item_action.append(last_co_ts - int(item_actions_before_co[-1]["timestamp"]))
        else:
            self.actions_since_last_item_action.append(last_co_step - 1)
            self.time_since_last_item_action.append(0)

    def update_same_impressions_in_session(self):
        same_impressions = filter(lambda x: x["impressions"] == self.current_session_clickouts[-1]["impressions"], self.current_session)
        same_impression_counter = len(list(same_impressions))
        self.same_impressions_in_session.append(same_impression_counter)

    def update_step(self):
        self.step.append(self.current_session_clickouts[-1]["step"])

    def update_last_action_type(self):
        if len(self.current_session) < 2:
            self.last_action_type.append("clickout item")
        else:
            self.last_action_type.append(self.current_session[-2]["action_type"])

    def update_price_above(self):
        self.price_above.append("|".join([str(self.current_prices[0])] + list(map(str, self.current_prices[:-1]))))

    def update_clickout_prob_time_position_offset(self):
        time_diff = self.clickout_item_item_last_timestamp[-1]
        if time_diff == DUMMY or time_diff is None or time_diff > 120:
            self.clickout_prob_time_position_offset.append("|".join(["0"]*len(self.current_impressions)))
        else:
            grouped_time_diff = group_time(time_diff)
            probs = [
                normalize_float(self.clickout_probs[(ind, grouped_time_diff)])
                for ind in range(len(self.current_impressions))
            ]
            self.clickout_prob_time_position_offset.append("|".join(map(str, probs)))

    def update_platform(self):
        self.platform.append(self.current_session_clickouts[-1]["platform"])

    def update_device(self):
        self.device.append(self.current_session_clickouts[-1]["device"])

    def update_active_filters(self):
        currently_active_filters = "|".join(
            [
                i["reference"]
                for i in filter(lambda x: x["action_type"] == "filter selection", self.current_session)
            ]
        )
        self.active_filters.append(currently_active_filters)

    def update_hour(self):
        self.hour.append(self.datetime.hour)

    def update_day_of_the_week(self):
        self.day_of_the_week.append(self.datetime.weekday())

    def update_cheapest(self):
        cheapest = min(self.current_prices)
        self.cheapest.append("|".join(["1" if pri == cheapest else "0" for pri in self.current_prices]))

    def update_most_expensive(self):
        exp = max(self.current_prices)
        self.most_expensive.append("|".join(["1" if pri == exp else "0" for pri in self.current_prices]))

    def update_timestamps(self):
        self.timestamps.append(self.timestamp)

    def update_impressions_max_price(self):
        self.impressions_max_price.append(max(self.current_prices))

    def update_impressions_min_price(self):
        self.impressions_min_price.append(min(self.current_prices))

    def update_impressions_price_range(self):
        self.impressions_price_range.append(
            self.impressions_max_price[-1] - self.impressions_min_price[-1]
        )

    def update_num_of_impressions(self):
        self.num_of_impressions.append(len(self.current_impressions))

    def update_price_rel_to_imp_min(self):
        imp_min = min(self.current_prices)

        self.price_rel_to_imp_min.append("|".join(
            [
                str(normalize_float(pri/imp_min)) for pri in self.current_prices
            ]
        ))

    def update_price_rel_to_imp_max(self):
        imp_max = max(self.current_prices)

        self.price_rel_to_imp_max.append("|".join(
            [
                str(normalize_float(pri / imp_max)) for pri in self.current_prices
            ]
        ))

    def update_price_rel_to_hist_min(self):
        pass

    def update_price_rel_to_hist_max(self):
        pass

    def update_historical_mean_rank(self):
        pass

    def update_rank_among_historical_mean_ranks(self):
        pass

    def update_price(self):
        self.price.append("|".join((str(x) for x in self.current_prices)))

    def update_item_id(self):
        self.item_id.append("|".join((str(x) for x in self.current_impressions)))

    def update_clicked(self):
        self.clicked.append("|".join(["1" if item_id == self.reference else "0" for item_id in self.current_impressions]))

    def update_city(self):
        self.city.append(self.current_city)

    def invalid_session_handler(self):
        self.session_id.append(self.current_session_id)
        for feat in self.feature_names[1:]:
            self.feature_array_map[feat].append(None)

    def run_updaters(self, feature_subset=None):
        if not self.current_session_valid:
            self.invalid_session_handler()
            return

        if feature_subset is not None:
            for feat in feature_subset:
                updater = self.feature_updater_map[feat]
                updater()

        else:
            for updater in self.feature_updater_map.values():
                updater()

    def extract_features(self):
        # print("Extracting features.")
        if not self.sorted:
            # print("Sorting started...")
            self.data_sorted = sorted(self.data, key=lambda x: (x["session_id"], int(x["timestamp"])))
            self.sorted = True
            del self.data
            # print("Sorting done...")

        for sess_id, sess in tqdm(groupby(self.data_sorted, lambda x: x["session_id"])):

            self.current_session_id = sess_id
            self.current_session = list(sess)
            self.current_session_clickouts = list(
                filter(lambda x: x["action_type"] == "clickout item", self.current_session))
            self.current_session_item_actions = list(
                filter(lambda x: (x["action_type"] in ITEM_ACTIONS and x["reference"] != "unknown"), self.current_session))
            if len(self.current_session_clickouts) == 0:
                self.invalid_session_ids.append(sess_id)
                self.current_session_valid = False

            for co in self.current_session_clickouts:
                if co["reference"] not in co["impressions"]:
                    self.invalid_session_ids.append(sess_id)
                    self.current_session_valid = False
                    break

            if self.current_session_valid:
                self.datetime = arrow.get(int(self.current_session_clickouts[-1]["timestamp"]))
                self.current_impressions = list(map(int, self.current_session_clickouts[-1]["impressions"].split("|")))
                self.current_prices = list(map(int, self.current_session_clickouts[-1]["prices"].split("|")))
                self.timestamp = self.current_session_clickouts[-1]["timestamp"]
                self.reference = int(self.current_session_clickouts[-1]["reference"])
                self.current_city = self.current_session_clickouts[-1]["city"]

            self.run_updaters()
            self.current_session_index += 1
            self.current_session_valid = True
        # print("extraction completed.")

    def validate_data(self):
        lengths = set()
        for feat in self.feature_names:
            lengths.add(len(self.feature_array_map[feat]))
            if len(lengths) != 1:
                print(f"A size inconsistency occured at {feat}")
                return 0
        return 1

    def save_features(self):
        if self.validate_data():
            as_df = pd.DataFrame(columns=self.feature_names)
            for feat, values in self.feature_array_map.items():
                as_df[feat] = values
            as_df.rename(columns={
                "timestamps": "timestamp",

            }).to_csv(self.write_path)

    def get_features(self):
        if self.validate_data():
            as_df = pd.DataFrame(columns=self.feature_names)
            for feat, values in self.feature_array_map.items():
                as_df[feat] = values
            return as_df.rename(columns={
                "timestamps": "timestamp",

            })


if __name__ == "__main__":
    csfe = SessionFeatures(events_sorted=True)
    csfe.extract_features()
    csfe.save_features()
