import os
import pandas as pd

ITEM_ACTIONS = {
    "interaction item image",
    "clickout item",
    "interaction item info",
    "interaction item deals",
    "interaction item rating",
}

EVENTS_PATH = os.path.join("data", "events_sorted.csv")
FULL_ITEM_FEATURES_PATH = os.path.join("data", "full_item_features.csv")
ITEM_DENSE_FEATURES_PATH = os.path.join("data", "item_dense_features.csv")
ITEM_POIS_PATH = os.path.join("data", "item_pois.joblib")
ITEM_PRICES_PATH = os.path.join("data", "item_prices.csv")
ITEM_SORT_BY_DIST_STATS_PATH = os.path.join("data", "item_sort_by_distance_stats.joblib")
ITEM_SORT_BY_POPULARITY_STATS_PATH = os.path.join("data", "item_sort_by_popularity_stats.joblib")
ITEM_SORT_BY_RATING_STATS_PATH = os.path.join("data", "item_sort_by_rating_stats.joblib")
PRICE_PCT_BY_CITY_PATH = os.path.join("data", "price_pct_by_city.joblib")
PRICE_PCT_BY_PLATFORM_PATH = os.path.join("data", "price_pct_by_platform.joblib")

FEATURE_DTYPES = {
    'session_id': str,
    'timestamp': int,
    'platform': str,
    'device': None,
    'active_filters': None,
    'hour': None,
    'day_of_the_week': None,
    'price': None,
    'item_id': None,
    'cheapest': None,
    'most_expensive': None,
    'impressions_max_price': None,
    'impressions_min_price': None,
    'impressions_price_range': None,
    'num_of_impressions': None,
    'price_rel_to_imp_min': None,
    'price_rel_to_imp_max': None,
    'rank': None,
    'price_rank': None,
    'price_rank_among_above': None,
    'session_start_ts': None,
    'price_vs_mean_price': None,
    'clickout_item_item_last_timestamp': None,
    'previous_click_price_diff_session': None,
    'top_of_impression': None,
    'num_of_item_actions': None,
    'price_rank_diff_last_interaction': None,
    'last_interaction_relative_position': None,
    'last_interaction_absolute_position': None,
    'actions_since_last_item_action': None,
    'time_since_last_item_action': None,
    'same_impression_in_session': None,
    'step': None,
    'last_action_type': None,
    'price_above': None,
    'clickout_prob_time_position_offset': None,
    'user_id': None,
    'user_start_ts': None,
    'sessions_of_user': None,
    'viewed_items_user': None,
    'item_clicked_before': None,
    'avg_price_similarity': None,
    'interacted_items_user': None,
    'viewed_items_avg_price': None,
    'last_price_diff_general': None,
    'interacted_items_avg_price': None,
    'viewed_items_avg_price_div': None,
    'viewed_items_avg_price_diff': None,
    'interacted_items_avg_price_div': None,
    'interacted_items_avg_price_diff': None,
    'interacted_and_viewed_items_price_diff': None,
}