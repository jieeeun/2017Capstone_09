from stock.Utils import get_max_min, get_mean_std

def normalize_data_maxmin(length, list_open, list_high, list_low, list_close):
    max_open, min_open      = get_max_min(list_open)
    max_high, min_high      = get_max_min(list_high)
    max_low, min_low        = get_max_min(list_low)
    max_close, min_close    = get_max_min(list_close)

    input_data, label_data = [], []
    for i in range(length):
        input_data.append((
            (list_open[i]   - min_open)     / (max_open     - min_open  + 1e-7),
            (list_high[i]   - min_high)     / (max_high     - min_high  + 1e-7),
            (list_low[i]    - min_low)      / (max_low      - min_low   + 1e-7),
            (list_close[i]  - min_close)    / (max_close    - min_close + 1e-7),
        ))
        label_data.append([(list_close[i] - min_close) / (max_close - min_close + 1e-7)])

    return (length, input_data, label_data)

def unormalize_data_maxmin(normalized_data, data):
    max_close, min_close = get_max_min(data)
    ret_list = [x * (max_close - min_close + 1e-7) + min_close for x in normalized_data]
    return ret_list

def normalize_data_meanstd(length, list_open, list_high, list_low, list_close):
    mean_open,  std_open    = get_mean_std(list_open)
    mean_high,  std_high    = get_mean_std(list_high)
    mean_low,   std_low     = get_mean_std(list_low)
    mean_close, std_close   = get_mean_std(list_close)

    input_data, label_data = [], []
    for i in range(length):
        input_data.append((
            (list_open[i]   - mean_open)     / std_open,
            (list_high[i]   - mean_high)     / std_high,
            (list_low[i]    - mean_low)      / std_low,
            (list_close[i]  - mean_close)    / std_close,
        ))
        label_data.append([(list_close[i]  - mean_close)    / std_close])

    return (length, input_data, label_data)

def unormalize_data_meanstd(normalized_data, data):
    mean_close, std_close = get_mean_std(data)
    ret_list = [x * std_close + mean_close for x in normalized_data]
    return ret_list

def normalize_data_percent(length, list_open, list_high, list_low, list_close):
    input_data, label_data = [], []
    for i in range(1, length):
        input_data.append((
            (list_open[i]   - list_close[i - 1]) / list_close[i - 1],
            (list_high[i]   - list_close[i - 1]) / list_close[i - 1],
            (list_low[i]    - list_close[i - 1]) / list_close[i - 1],
            (list_close[i]  - list_close[i - 1]) / list_close[i - 1],
        ))
        label_data.append([(list_close[i] - list_close[i - 1]) / list_close[i - 1]])
    return (length-1, input_data, label_data)

def unormalize_data_percent(normalized_data, data):
    return normalized_data