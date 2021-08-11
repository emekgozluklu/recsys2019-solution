import pandas as pd
import os

DENSE_FEATURES_WRITE_PATH = os.path.join("data", "item_dense_features.csv")
ITEM_FEATURES_FILEPATH = os.path.join("data", "full_item_features.csv")

item_features = pd.read_csv(ITEM_FEATURES_FILEPATH)


def normalize_feature_name(name):
    return name.replace(" ", "_").lower()


hotel_categories = {
    "Hotel": "hotel",
    "Resort": "resort",
    "Hostal (ES)": "hostal",
    "Motel": "motel",
    "House / Apartment": "house",
}

# important features are taken from the paper of Logic AI team
# link: https://github.com/logicai-io/recsys2019/
important_features = [
    "Free WiFi (Combined)",
    "Swimming Pool (Combined Filter)",
    "Car Park",
    "Serviced Apartment",
    "Air Conditioning",
    "Spa (Wellness Facility)",
    "Pet Friendly",
    "All Inclusive (Upon Inquiry)",
]

item_features["n_properties"] = item_features.drop(columns=["item_index", "item_id"]).sum(axis=1)

item_features["rating"] = None
item_features.loc[item_features["Satisfactory Rating"] == 1, "rating"] = 1
item_features.loc[item_features["Good Rating"] == 1, "rating"] = 2
item_features.loc[item_features["Very Good Rating"] == 1, "rating"] = 3
item_features.loc[item_features["Excellent Rating"] == 1, "rating"] = 4


item_features["stars"]= None
item_features.loc[item_features["1 Star"] == 1, "stars"] = 1
item_features.loc[item_features["2 Star"] == 1, "stars"] = 2
item_features.loc[item_features["3 Star"] == 1, "stars"] = 3
item_features.loc[item_features["4 Star"] == 1, "stars"] = 4
item_features.loc[item_features["5 Star"] == 1, "stars"] = 5

item_features["hotel_category"] = None
item_features.loc[item_features["Hotel"] == 1, "hotel_category"] = "hotel"
item_features.loc[item_features["Resort"] == 1, "hotel_category"] = "resort"
item_features.loc[item_features["Hostal (ES)"] == 1, "hotel_category"] = "hostel"
item_features.loc[item_features["Motel"] == 1, "hotel_category"] = "motel"
item_features.loc[item_features["House / Apartment"] == 1, "hotel_category"] = "house"

final_features = ["item_index", "item_id"] + important_features + ["rating", "stars", "hotel_category"]
item_features = item_features[final_features]

item_features.rename(
    columns=dict(zip(
        item_features.columns,
        map(normalize_feature_name, item_features.columns))),
    inplace=True
)

item_features.to_csv(DENSE_FEATURES_WRITE_PATH)
print("Dense features extracted and saved.")
