import os
from src.sampling import data_sampler
from src.preparation import init_data_preparation, generate_click_indices, item_data_parser, item_poi_mapper,\
    item_price_percentiles_extractor, item_prices_parser, item_dense_feature_extractor, item_ratings_extractor
from src.extraction import session_feature_extractor, item_feature_extractor, user_feature_extractor,\
    merge_session_and_user_features
from src.model import normalize_features


def run_sampler(outfile="train_sample.csv", n_sessions=50000):
    os.chdir("sampling")
    data_sampler.sample_data(outfile, n_sessions)
    os.chdir("..")


def run_data_preparation(input_file="train_sample.csv"):
    print("#"*50)
    print("DATA PREPARATION")
    os.chdir("preparation")
    init_data_preparation.run(read_path=input_file)
    print("Data preparation initialized... [1 / 8]")
    generate_click_indices.run()
    print("Click indices generated... [2 / 8]")
    item_data_parser.run()
    print("Parsed item data... [3 / 8]")
    item_poi_mapper.run()
    print("Items mapped to POIs... [4 / 8]")
    item_price_percentiles_extractor.run()
    print("Item price percentiles calculated... [5 / 8]")
    item_prices_parser.run()
    print("Item prices parsed... [6 / 8]")
    item_dense_feature_extractor.run()
    print("Item dense features extracted... [7 / 8]")
    item_ratings_extractor.run()
    print("Item ratings extracted... [8 / 8]")
    print("Data preperation done.")
    print("#"*50)
    os.chdir("..")


def run_feature_extraction():
    os.chdir("extraction")
    print("#"*50)
    print("FEATURE EXTRACTION")

    sfe = session_feature_extractor.SessionFeatures(events_sorted=False)
    sfe.extract_features()
    sf = sfe.get_features()
    sfe.save_features()
    print("Session features extracted. [1/4]")

    ufe = user_feature_extractor.UserFeatures(events_sorted=False)
    ufe.extract_features()
    uf = ufe.get_features()
    ufe.save_features()
    print("User features extracted. [2/4]")

    merge_session_and_user_features.merge(session_features=sf, user_features=uf)
    print("Session and User features merged. [3/4]")

    ife = item_feature_extractor.ItemFeatures(events_sorted=False)
    ife.extract_features()
    ife.save_features()
    print("Item features extracted, combined and saved. [4/4]")
    os.chdir("..")


def run_model():
    os.chdir("model")
    print("#"*50)
    print("MODEL")

    dp = normalize_features.DataPreprocessing()
    dp.run()
    dp.save()
    print("Feature normalized [1 / X]")
    os.chdir("..")


if __name__ == "__main__":
    run_sampler("train_sample.csv", 50000)
    run_data_preparation("../../data/sampled/train_sample.csv")
    run_feature_extraction()
    run_model()
