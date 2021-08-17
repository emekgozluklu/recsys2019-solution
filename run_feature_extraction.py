from item_feature_extractor import ItemFeatures
from session_feature_extractor import SessionFeatures
from user_feature_extractor import UserFeatures
import merge_features
import time
import pandas as pd


def run_feature_extraction(save_separately=False,
                           read_only=False,
                           session_file="data/events_sorted.csv",
                           user_file="data/events_sorted.csv",
                           item_file="data/user_session_merged.csv",
                           save_as="data/all_features.csv"):

    t = time.time()
    if read_only:
        uf = pd.read_csv("data/user_features.csv", index_col=0)
        sf = pd.read_csv("data/session_features.csv", index_col=0)

    else:
        print("Extracting Session Features")
        sfe = SessionFeatures(session_file, events_sorted=False)
        sfe.extract_features()
        sf = sfe.get_features()
        print("Done!")

        print("Extracting User Features")
        ufe = UserFeatures(user_file, events_sorted=False)
        ufe.extract_features()
        uf = ufe.get_features()
        print("Done!")

        print(uf.columns)
        print(sf.columns)

        if save_separately:
            ufe.save_features()
            sfe.save_features()

    merge_features.merge_session_and_user_features(session_features=sf, user_features=uf)

    ife = ItemFeatures(item_file, events_sorted=False)
    ife.extract_features()
    all_feats = ife.get_features()

    all_feats.to_csv(save_as)
    print("Total time: ", time.time() - t, " seconds")


if __name__ == "__main__":
    run_feature_extraction(read_only=False)
