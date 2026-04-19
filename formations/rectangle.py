from typing import Dict, Optional, List

import pandas as pd

INCLUDE_RECTANGLE_BREAKOUT_DOWN = False


def _build_touch_clusters(mask: pd.Series) -> List[tuple[int, int]]:
    idx_positions = [i for i, is_touch in enumerate(mask.tolist()) if bool(is_touch)]
    if not idx_positions:
        return []

    clusters: List[tuple[int, int]] = []
    cluster_start = idx_positions[0]
    cluster_end = idx_positions[0]

    for pos in idx_positions[1:]:
        if pos == (cluster_end + 1):
            cluster_end = pos
        else:
            clusters.append((cluster_start, cluster_end))
            cluster_start = pos
            cluster_end = pos

    clusters.append((cluster_start, cluster_end))
    return clusters


def _count_spaced_touch_clusters(mask: pd.Series, min_gap: int) -> int:
    clusters = _build_touch_clusters(mask)
    if not clusters:
        return 0

    count = 1
    last_cluster_end = clusters[0][1]

    for cluster_start, cluster_end in clusters[1:]:
        if (cluster_start - last_cluster_end) >= min_gap:
            count += 1
            last_cluster_end = cluster_end

    return count


def _check_rectangle_breakout_on_df(
        df: pd.DataFrame,
        touch_tolerance_of_height: float = 0.15,
        min_touches: int = 2,
        breakout_pct: float = 0.0,
        max_height_pct: float = 0.10,
        min_days_between_touches_ratio: float = 0.25,
        length_days_values: tuple[int, ...] = (20, 35, 50, 65, 80, 180),
) -> Optional[str]:
    required_columns = {"Close"}
    if df.empty or not required_columns.issubset(df.columns):
        return None

    available_price_columns = [col for col in ["Close", "High", "Low"] if col in df.columns]
    df = df[available_price_columns].dropna().sort_index()
    if len(df) < 21:
        return None

    close_today = float(df.iloc[-1]["Close"])
    history = df.iloc[:-1]

    use_high_low_for_touches = {"High", "Low"}.issubset(history.columns)

    up_matches: List[dict] = []
    down_matches: List[dict] = []

    for length_days in length_days_values:
        if len(history) < length_days:
            continue

        min_days_between_touches = max(1, int(round(length_days * min_days_between_touches_ratio)))

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

        if use_high_low_for_touches:
            support_mask = window["Low"] <= (support + tol)
            resistance_mask = window["High"] >= (resistance - tol)
        else:
            support_mask = window["Close"] <= (support + tol)
            resistance_mask = window["Close"] >= (resistance - tol)

        touches_support = _count_spaced_touch_clusters(support_mask, min_days_between_touches)
        touches_resistance = _count_spaced_touch_clusters(resistance_mask, min_days_between_touches)

        if touches_support < min_touches or touches_resistance < min_touches:
            continue

        breakout_up_level = resistance * (1.0 + breakout_pct)
        breakout_down_level = support * (1.0 - breakout_pct)

        match = {
            "length_days": length_days,
            "support": support,
            "resistance": resistance,
            "touches_support": touches_support,
            "touches_resistance": touches_resistance,
            "height_pct": height_pct,
        }

        if close_today > breakout_up_level:
            up_matches.append(match)
        elif INCLUDE_RECTANGLE_BREAKOUT_DOWN and close_today < breakout_down_level:
            down_matches.append(match)

    parts: List[str] = []

    if up_matches:
        representative = max(up_matches, key=lambda x: x["length_days"])
        windows = sorted(match["length_days"] for match in up_matches)
        windows_str = ",".join(str(x) for x in windows)
        parts.append(
            f"🟩(rectangle) ⬆️ count={len(up_matches)} windows={windows_str} "
            f"({representative['support']:.2f}↔️{representative['resistance']:.2f}, "
            f"h={representative['height_pct'] * 100:.1f}%)"
        )

    if down_matches:
        representative = max(down_matches, key=lambda x: x["length_days"])
        windows = sorted(match["length_days"] for match in down_matches)
        windows_str = ",".join(str(x) for x in windows)
        parts.append(
            f"🟩(rectangle) ⬇️ count={len(down_matches)} windows={windows_str} "
            f"({representative['support']:.2f}↔️{representative['resistance']:.2f}, "
            f"h={representative['height_pct'] * 100:.1f}%)"
        )

    return "   ".join(parts) if parts else None


def check_rectangle_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        touch_tolerance_of_height: float = 0.15,
        min_touches: int = 2,
        breakout_pct: float = 0.0,
        max_height_pct: float = 0.10,
        min_days_between_touches_ratio: float = 0.25,
        length_days_values: tuple[int, ...] = (20, 35, 50, 65, 80, 180),
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_rectangle_breakout_on_df(
        df=df,
        touch_tolerance_of_height=touch_tolerance_of_height,
        min_touches=min_touches,
        breakout_pct=breakout_pct,
        max_height_pct=max_height_pct,
        min_days_between_touches_ratio=min_days_between_touches_ratio,
        length_days_values=length_days_values,
    )