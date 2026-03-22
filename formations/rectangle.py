from typing import Dict, Optional

import pandas as pd

INCLUDE_RECTANGLE_BREAKOUT_DOWN = False


def check_rectangle_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        touch_tolerance_of_height: float = 0.15
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index")[["Close"]].dropna().sort_index()
    if len(df) < 21:
        return None

    close_today = float(df.iloc[-1]["Close"])
    history = df.iloc[:-1]  # D-... do wczoraj

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

    min_touches = 2
    parts: list[str] = []

    for length_days in list(range(20, 91, 15)) + [180]:
        if len(history) < length_days:
            continue

        max_height_pct = 0.10 if length_days <= 40 else 0.20
        min_days_between_touches = max(1, length_days // 4)

        window = history.iloc[-length_days:]
        support = float(window["Close"].min())
        resistance = float(window["Close"].max())

        if support <= 0:
            continue

        height = resistance - support
        if height <= 0:
            continue

        height_pct = height / support
        if height_pct > max_height_pct:
            continue

        tol = touch_tolerance_of_height * height
        support_mask = window["Close"] <= (support + tol)
        resistance_mask = window["Close"] >= (resistance - tol)

        touches_support = count_spaced_touches(support_mask, min_days_between_touches)
        touches_resistance = count_spaced_touches(resistance_mask, min_days_between_touches)

        if touches_support < min_touches or touches_resistance < min_touches:
            continue

        if close_today > resistance:
            parts.append(f"▭{length_days}⬆️ ({support:.2f}↔️{resistance:.2f})")
        elif INCLUDE_RECTANGLE_BREAKOUT_DOWN and close_today < support:
            parts.append(f"▭{length_days}⬇️ ({support:.2f}↔️{resistance:.2f})")

    return "   ".join(parts) if parts else None
