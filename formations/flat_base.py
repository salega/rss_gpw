from typing import Dict, Optional, List

import pandas as pd

INCLUDE_BASE_BREAKOUT_DOWN = False


def _check_flat_base_breakout_on_df(
        df: pd.DataFrame,
        touch_tolerance_pct: float = 0.005,
        min_touches_resistance: int = 3,
        min_breakout_pct: float = 0.01,
        max_base_depth_pct: float = 0.08,
        use_low_for_depth: bool = False,
        min_close_near_resistance_ratio: float = 0.9,
        near_resistance_pct: float = 0.04,
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

    def count_spaced_touches(mask: pd.Series, min_gap: int) -> int:
        idx_positions = [i for i, is_touch in enumerate(mask.tolist()) if bool(is_touch)]
        if not idx_positions:
            return 0

        count = 1
        last_pos = idx_positions[0]

        for pos in idx_positions[1:]:
            if (pos - last_pos) >= min_gap:
                count += 1
                last_pos = pos

        return count

    parts: list[str] = []

    for length_days in length_days_candidates:
        if len(history) < length_days:
            continue

        window = history.iloc[-length_days:]
        resistance = float(window["Close"].max())
        if resistance <= 0:
            continue

        base_low = float(window["Low"].min()) if use_low_for_depth else float(window["Close"].min())
        base_depth_pct = (resistance - base_low) / resistance
        if base_depth_pct > max_base_depth_pct:
            continue

        near_resistance_level = resistance * (1.0 - near_resistance_pct)
        close_near_resistance_ratio = float((window["Close"] >= near_resistance_level).mean())
        if close_near_resistance_ratio < min_close_near_resistance_ratio:
            continue

        tol_abs = touch_tolerance_pct * resistance
        breakout_abs = min_breakout_pct * resistance
        resistance_mask = window["Close"] >= (resistance - tol_abs)

        min_days_between_touches = max(1, length_days // 4)
        touches_resistance = count_spaced_touches(resistance_mask, min_days_between_touches)
        if touches_resistance < min_touches_resistance:
            continue

        if close_today > (resistance + breakout_abs):
            parts.append(f"💥(flat base) {length_days}⬆️ (R≈{resistance:.2f}, touches={touches_resistance})")
        elif INCLUDE_BASE_BREAKOUT_DOWN and close_today < (resistance - breakout_abs):
            parts.append(f"💥(flat base) {length_days}⬇️ (R≈{resistance:.2f}, touches={touches_resistance})")

    return "   ".join(parts) if parts else None


def check_flat_base_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        touch_tolerance_pct: float = 0.01,
        min_touches_resistance: int = 3,
        min_breakout_pct: float = 0.01,
        max_base_depth_pct: float = 0.08,
        use_low_for_depth: bool = False,
        min_close_near_resistance_ratio: float = 0.9,
        near_resistance_pct: float = 0.04
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_flat_base_breakout_on_df(
        df=df,
        touch_tolerance_pct=touch_tolerance_pct,
        min_touches_resistance=min_touches_resistance,
        min_breakout_pct=min_breakout_pct,
        max_base_depth_pct=max_base_depth_pct,
        use_low_for_depth=use_low_for_depth,
        min_close_near_resistance_ratio=min_close_near_resistance_ratio,
        near_resistance_pct=near_resistance_pct
    )
