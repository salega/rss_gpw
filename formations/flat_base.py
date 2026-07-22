from typing import Dict, Optional, List

import pandas as pd

INCLUDE_BASE_BREAKOUT_DOWN = False


def _check_flat_base_breakout_on_df(
        df: pd.DataFrame,
        min_breakout_pct: float = 0.01,
        max_center_deviation_pct: float = 0.03,
        use_low_for_depth: bool = False,
        base_length_days_list: Optional[List[int]] = None,
) -> Optional[str]:
    required_cols = {"Close"}
    if use_low_for_depth:
        required_cols.add("Low")

    if df.empty or not required_cols.issubset(set(df.columns)):
        return None

    selected_cols = ["Close", "Low"] if use_low_for_depth else ["Close"]
    df = df[selected_cols].dropna().sort_index()

    length_days_candidates = base_length_days_list or (list(range(20, 91, 15)) + [180])
    length_days_candidates = sorted({int(length_days) for length_days in length_days_candidates if int(length_days) > 0})
    if not length_days_candidates:
        return None

    min_length_days = min(length_days_candidates)
    if len(df) < min_length_days + 1:
        return None

    close_today = float(df.iloc[-1]["Close"])
    history = df.iloc[:-1]

    if len(history) < min_length_days:
        return None

    parts: list[str] = []

    for length_days in length_days_candidates:
        if len(history) < length_days:
            continue

        window = history.iloc[-length_days:]
        center_value = float(window["Close"].median())
        if center_value <= 0:
            continue

        upper_reference = float(window["High"].max()) if use_low_for_depth and "High" in window.columns else float(window["Close"].max())
        lower_reference = float(window["Low"].min()) if use_low_for_depth else float(window["Close"].min())

        upper_deviation_pct = (upper_reference - center_value) / center_value
        lower_deviation_pct = (center_value - lower_reference) / center_value
        max_observed_deviation_pct = max(upper_deviation_pct, lower_deviation_pct)
        if max_observed_deviation_pct > max_center_deviation_pct:
            continue

        breakout_level = upper_reference * (1.0 + min_breakout_pct)

        if close_today > breakout_level:
            parts.append(
                f"💥(flat base) {length_days}⬆️ (X≈{center_value:.2f}, dev≈{max_observed_deviation_pct * 100:.1f}%)"
            )
        elif INCLUDE_BASE_BREAKOUT_DOWN and close_today < (lower_reference * (1.0 - min_breakout_pct)):
            parts.append(
                f"💥(flat base) {length_days}⬇️ (X≈{center_value:.2f}, dev≈{max_observed_deviation_pct * 100:.1f}%)"
            )

    return "   ".join(parts) if parts else None


def check_flat_base_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        min_breakout_pct: float = 0.01,
        max_center_deviation_pct: float = 0.02,
        use_low_for_depth: bool = True
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_flat_base_breakout_on_df(
        df=df,
        min_breakout_pct=min_breakout_pct,
        max_center_deviation_pct=max_center_deviation_pct,
        use_low_for_depth=use_low_for_depth,
        base_length_days_list=[20],
    )
