import pandas as pd
from typing import Dict, Optional

INCLUDE_BASE_BREAKOUT_DOWN = False


def check_flat_base_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        touch_tolerance_pct: float = 0.01,
        min_touches_resistance: int = 3
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index")[["Close"]].dropna().sort_index()
    if len(df) < 21:
        return None

    close_today = float(df.iloc[-1]["Close"])
    history = df.iloc[:-1]  # do wczoraj

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

    for length_days in list(range(20, 91, 15)) + [180]:
        if len(history) < length_days:
            continue

        window = history.iloc[-length_days:]
        resistance = float(window["Close"].max())
        if resistance <= 0:
            continue

        tol_abs = touch_tolerance_pct * resistance
        resistance_mask = window["Close"] >= (resistance - tol_abs)

        min_days_between_touches = max(1, length_days // 4)
        touches_resistance = count_spaced_touches(resistance_mask, min_days_between_touches)
        if touches_resistance < min_touches_resistance:
            continue

        if close_today > resistance:
            parts.append(f"▱{length_days}⬆️ (R≈{resistance:.2f}, touches={touches_resistance})")
        elif INCLUDE_BASE_BREAKOUT_DOWN and close_today < (resistance - tol_abs):
            parts.append(f"▱{length_days}⬇️ (R≈{resistance:.2f}, touches={touches_resistance})")

    return "   ".join(parts) if parts else None