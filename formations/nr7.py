import pandas as pd
from typing import Dict, Optional


INCLUDE_DOWN_NR7 = False


def check_nr7_confirmed_today(prices: Dict[pd.Timestamp, Dict[str, float]]) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index")[["High", "Low", "Close"]].dropna().sort_index()

    if len(df) < 8:
        return None

    confirm_candle = df.iloc[-1]  # D
    window_7 = df.iloc[-8:-1]  # D-7..D-1
    last_candle = df.iloc[-2]  # D-1

    ranges = (window_7["High"] - window_7["Low"])
    last_range = float(last_candle["High"] - last_candle["Low"])

    # NR7: D-1 ma najwęższy zakres z 7 świec
    if last_range != float(ranges.min()):
        return None

    # remisy: jeśli kilka dni ma taki sam minimalny zakres, nie uznajemy NR7
    if int((ranges == ranges.min()).sum()) != 1:
        return None

    high_7 = float(window_7["High"].max())
    low_7 = float(window_7["Low"].min())
    close_d = float(confirm_candle["Close"])

    if close_d > high_7:
        return "UP"

    if INCLUDE_DOWN_NR7 and close_d < low_7:
        return "DOWN"

    return None