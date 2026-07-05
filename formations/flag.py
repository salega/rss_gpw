import pandas as pd
from typing import Dict, Optional, List


def _prepare_flag_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Close", "High", "Low"]
    has_open = "Open" in df.columns

    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    if has_open:
        return df[["Open", "Close", "High", "Low"]].dropna().sort_index()

    return df[["Close", "High", "Low"]].dropna().sort_index()


def _build_signal(pole_start_date: pd.Timestamp, pole_peak_date: pd.Timestamp, max_price: float) -> str:
    pole_start_str = pole_start_date.strftime("%Y-%m-%d")
    pole_peak_str = pole_peak_date.strftime("%Y-%m-%d")
    return f"🚩{pole_start_str} → {pole_peak_str} ({max_price:.2f})"


def _get_candle_body_high_low(
        has_open: bool,
        open_price: Optional[float],
        close_price: float,
        prev_close: float,
) -> tuple[float, float]:
    if has_open and open_price is not None:
        return max(open_price, close_price), min(open_price, close_price)

    return max(prev_close, close_price), min(prev_close, close_price)


def _find_pole_from_index(
        working_df: pd.DataFrame,
        start_idx: int,
        pole_min_days: int,
        pole_max_days: int,
        pole_min_growth: float,
        pole_max_daily_decline: float,
        max_days_without_new_high: int,
) -> Optional[dict]:
    has_open = "Open" in working_df.columns
    n = len(working_df)

    if start_idx >= n - pole_min_days:
        return None

    if start_idx > 0:
        first_close = float(working_df.iloc[start_idx]["Close"])
        day_before_close = float(working_df.iloc[start_idx - 1]["Close"])
        if first_close <= day_before_close:
            return None

    current_max = float(
        working_df.iloc[start_idx]["Close"] if has_open else working_df.iloc[start_idx]["High"]
    )
    actual_pole_end_idx = start_idx
    days_without_new_high = 0

    closes: List[float] = []

    max_end_idx = min(n - 1, start_idx + pole_max_days - 1)

    for idx in range(start_idx, max_end_idx + 1):
        row = working_df.iloc[idx]
        current_close = float(row["Close"])
        current_peak = float(row["Close"] if has_open else row["High"])
        closes.append(current_close)

        if current_peak > current_max:
            current_max = current_peak
            actual_pole_end_idx = idx
            days_without_new_high = 0
        else:
            days_without_new_high += 1

        local_i = idx - start_idx

        if days_without_new_high > max_days_without_new_high:
            break

        if local_i > 0 and current_close < closes[local_i - 1]:
            if local_i < 2:
                return None

            day_before_prev_close = closes[local_i - 2]
            previous_close = closes[local_i - 1]
            previous_day_gain = previous_close - day_before_prev_close

            if previous_day_gain <= 0:
                return None

            decline = previous_close - current_close
            max_allowed_decline = previous_day_gain * pole_max_daily_decline

            if decline > max_allowed_decline:
                return None

    actual_pole_length = actual_pole_end_idx - start_idx + 1
    if actual_pole_length < pole_min_days:
        return None

    pole_start_price = float(working_df.iloc[start_idx]["Close"])
    pole_end_price = float(working_df.iloc[actual_pole_end_idx]["Close"])

    if pole_start_price <= 0:
        return None

    pole_growth = (pole_end_price - pole_start_price) / pole_start_price
    if pole_growth < pole_min_growth:
        return None

    max_price = current_max
    pole_height = max_price - pole_start_price
    if pole_height <= 0:
        return None

    return {
        "pole_start_idx": start_idx,
        "pole_end_idx": actual_pole_end_idx,
        "pole_start_date": working_df.index[start_idx],
        "pole_peak_date": working_df.index[actual_pole_end_idx],
        "pole_start_price": float(pole_start_price),
        "pole_end_price": float(pole_end_price),
        "pole_growth": float(pole_growth),
        "max_price": float(max_price),
        "pole_height": float(pole_height),
    }


def _find_breakout_after_pole(
        working_df: pd.DataFrame,
        pole: dict,
        flag_min_days: int,
        flag_max_days_until_breakout: int,
        flag_max_retracement: float,
) -> Optional[dict]:
    has_open = "Open" in working_df.columns
    n = len(working_df)

    pole_end_idx = int(pole["pole_end_idx"])
    pole_start_price = float(pole["pole_start_price"])
    max_price = float(pole["max_price"])
    pole_height = float(pole["pole_height"])

    half_pole = pole_start_price + (pole_height / 2.0)

    flag_start_idx = pole_end_idx + 1
    if flag_start_idx >= n:
        return None

    flag_candle_highs: List[float] = []
    flag_candle_lows: List[float] = []

    for idx in range(flag_start_idx, min(n, pole_end_idx + 1 + flag_max_days_until_breakout + 1)):
        close_price = float(working_df.iloc[idx]["Close"])

        if idx == flag_start_idx:
            prev_close = float(working_df.iloc[pole_end_idx]["Close"])
        else:
            prev_close = float(working_df.iloc[idx - 1]["Close"])

        open_price = float(working_df.iloc[idx]["Open"]) if has_open else None

        candle_high, candle_low = _get_candle_body_high_low(
            has_open=has_open,
            open_price=open_price,
            close_price=close_price,
            prev_close=prev_close,
        )

        days_in_flag = idx - flag_start_idx + 1

        if candle_high > max_price:
            if days_in_flag < flag_min_days:
                return None

            flag_end_idx = idx - 1
            breakout_idx = idx

            if flag_end_idx < flag_start_idx:
                return None

            if len(flag_candle_lows) < flag_min_days:
                return None

            if any(h > max_price for h in flag_candle_highs):
                return None

            if any(low < half_pole for low in flag_candle_lows):
                return None

            flag_low = min(flag_candle_lows)
            retracement = (max_price - flag_low) / pole_height
            if retracement > flag_max_retracement:
                return None

            score = float(pole["pole_growth"]) * (1 - retracement) * (
                    1 - len(flag_candle_lows) / flag_max_days_until_breakout
            )

            return {
                **pole,
                "flag_start_idx": flag_start_idx,
                "flag_end_idx": flag_end_idx,
                "breakout_idx": breakout_idx,
                "breakout_date": working_df.index[breakout_idx],
                "score": float(score),
            }

        flag_candle_highs.append(candle_high)
        flag_candle_lows.append(candle_low)

        if candle_low < half_pole:
            return None

        flag_low = min(flag_candle_lows)
        retracement = (max_price - flag_low) / pole_height
        if retracement > flag_max_retracement:
            return None

    return None


def find_flag_breakouts_on_df(
        df: pd.DataFrame,
        pole_min_days: int = 4,
        pole_max_days: int = 15,
        pole_min_growth: float = 0.06,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 20,
        flag_max_retracement: float = 0.33,
) -> List[dict]:
    working_df = _prepare_flag_df(df)
    if working_df.empty:
        return []

    min_required_len = pole_min_days + flag_min_days + 1
    if len(working_df) < min_required_len:
        return []

    results: List[dict] = []
    i = 0
    n = len(working_df)

    while i < n - min_required_len + 1:
        pole = _find_pole_from_index(
            working_df=working_df,
            start_idx=i,
            pole_min_days=pole_min_days,
            pole_max_days=pole_max_days,
            pole_min_growth=pole_min_growth,
            pole_max_daily_decline=pole_max_daily_decline,
            max_days_without_new_high=max_days_without_new_high,
        )

        if pole is None:
            i += 1
            continue

        breakout = _find_breakout_after_pole(
            working_df=working_df,
            pole=pole,
            flag_min_days=flag_min_days,
            flag_max_days_until_breakout=flag_max_days_until_breakout,
            flag_max_retracement=flag_max_retracement,
        )

        if breakout is None:
            i += 1
            continue

        signal = _build_signal(
            pole_start_date=breakout["pole_start_date"],
            pole_peak_date=breakout["pole_peak_date"],
            max_price=float(breakout["max_price"]),
        )

        results.append(
            {
                "date": breakout["breakout_date"],
                "signal": signal,
                "pole_start_date": breakout["pole_start_date"],
                "pole_peak_date": breakout["pole_peak_date"],
                "max_price": float(breakout["max_price"]),
                "score": float(breakout["score"]),
            }
        )

        i = int(breakout["breakout_idx"]) + 1

    return results


def _check_flag_breakout_on_df(
        df: pd.DataFrame,
        pole_min_days: int = 4,
        pole_max_days: int = 15,
        pole_min_growth: float = 0.06,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 20,
        flag_max_retracement: float = 0.33,
        breakout_idx: Optional[int] = None,
) -> Optional[str]:
    working_df = _prepare_flag_df(df)
    if working_df.empty:
        return None

    if breakout_idx is not None:
        if breakout_idx < 0 or breakout_idx >= len(working_df):
            return None
        working_df = working_df.iloc[:breakout_idx + 1]

    found = find_flag_breakouts_on_df(
        df=working_df,
        pole_min_days=pole_min_days,
        pole_max_days=pole_max_days,
        pole_min_growth=pole_min_growth,
        pole_max_daily_decline=pole_max_daily_decline,
        max_days_without_new_high=max_days_without_new_high,
        flag_min_days=flag_min_days,
        flag_max_days_until_breakout=flag_max_days_until_breakout,
        flag_max_retracement=flag_max_retracement,
    )

    if not found:
        return None

    last_signal = found[-1]
    if pd.Timestamp(last_signal["date"]) != working_df.index[-1]:
        return None

    return str(last_signal["signal"])


def check_flag_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        pole_min_days: int = 4,
        pole_max_days: int = 15,
        pole_min_growth: float = 0.06,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 20,
        flag_max_retracement: float = 0.50
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index")
    return _check_flag_breakout_on_df(
        df=df,
        pole_min_days=pole_min_days,
        pole_max_days=pole_max_days,
        pole_min_growth=pole_min_growth,
        pole_max_daily_decline=pole_max_daily_decline,
        max_days_without_new_high=max_days_without_new_high,
        flag_min_days=flag_min_days,
        flag_max_days_until_breakout=flag_max_days_until_breakout,
        flag_max_retracement=flag_max_retracement,
    )