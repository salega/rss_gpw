from operator import itemgetter
from typing import Dict, Tuple, Optional

import pandas as pd


def get_max_value(prices):
    if not prices:
        return None
    return max(prices.items(), key=itemgetter(1))


def check_if_price_above_emas(data: Dict[pd.Timestamp, float]) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if not data:
        raise ValueError("Pusty zbiór danych")

    s = pd.Series(data).sort_index()
    last_price = s.iloc[-1]

    def ema_above(span: int) -> Optional[bool]:
        ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
        if pd.isna(ema.iloc[-1]):
            return None
        return bool(last_price > ema.iloc[-1])

    return (
        ema_above(8),
        ema_above(30),
        ema_above(200),
    )
