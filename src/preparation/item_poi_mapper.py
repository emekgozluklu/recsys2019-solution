import joblib
import pandas as pd
import numpy as np
from collections import defaultdict


def run():
    events = pd.read_csv("../../data/events_sorted.csv", index_col=0)
    poi_events = events[events["action_type"] == "search for poi"]

    item_pois = defaultdict(set)

    for ind, row in poi_events.iterrows():
        if row["fake_impressions"] is not np.NaN:
            imps = str(row["fake_impressions"]).split("|")
            poi = row["reference"]
            for imp in imps:
                item_pois[int(imp)].add(poi)

    print("Items mapped with POIs.")
    joblib.dump(item_pois, "../../data/item_to_poi_map.joblib")


if __name__ == "__main__":
    run()
