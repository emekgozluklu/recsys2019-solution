import os

ITEM_ACTIONS = {
    "interaction item image",
    "clickout item",
    "interaction item info",
    "interaction item deals",
    "interaction item rating",
}

EVENTS_PATH = os.path.join("data", "events.csv")
FULL_ITEM_FEATURES_PATH = os.path.join("data", "full_item_features.csv")
ITEM_DENSE_FEATURES_PATH = os.path.join("data", "item_dense_features.csv")
ITEM_POIS_PATH = os.path.join("data", "item_pois.joblib")
ITEM_PRICES_PATH = os.path.join("data", "item_prices.csv")
ITEM_SORT_BY_DIST_STATS_PATH = os.path.join("data", "item_sort_by_distance_stats.joblib")
ITEM_SORT_BY_POPULARITY_STATS_PATH = os.path.join("data", "item_sort_by_popularity_stats.joblib")
ITEM_SORT_BY_RATING_STATS_PATH = os.path.join("data", "item_sort_by_rating_stats.joblib")
PRICE_PCT_BY_CITY_PATH = os.path.join("data", "price_pct_by_city.joblib")
PRICE_PCT_BY_PLATFORM_PATH = os.path.join("data", "price_pct_by_platform.joblib")
