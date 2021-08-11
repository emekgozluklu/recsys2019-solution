def get_price_ranks_from_prices_list(prices):
    prices_sorted = prices.copy()
    prices_sorted.sort()

    ranks = {}

    rank = 1

    for index in range(len(prices_sorted)):
        element = prices_sorted[index];

        if element not in ranks:
            ranks[element] = rank
            rank += 1

    ranks_arr = [None] * len(prices)
    for index in range(len(prices)):
        ranks_arr[index] = ranks[prices[index]]
    return ranks_arr


def get_hotel_price_by_id(hotel_id, impressions, prices):
    impressions = list(map(int, impressions.split("|")))
    prices = list(map(int, prices.split("|")))

    try:
        ind = impressions.index(int(hotel_id))
    except ValueError:
        print(impressions, hotel_id)
        return -1
    return prices[ind]


def dummy_function():
    pass


def clickout_filter(x):
    return x["action_type"] == "clickout item"

def get_dataframe_memory_usage(df):
    return df.memory_usage(deep=True).sum() * 10**(-9)
