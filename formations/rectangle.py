from typing import Dict, Optional, List

import pandas as pd

INCLUDE_RECTANGLE_BREAKOUT_DOWN = False

RECTANGLE_DAILY_SCAN_CONFIGS = (
    {
        "name": "R1-balanced",
        "touch_tolerance_of_height": 0.15,
        "min_touches": 2,
        "breakout_pct": 0.01,
        "max_height_pct": 0.20,
        "min_days_between_touches_ratio": 0.25,
        "length_days_values": (35,),
        "historical_stats": {
            "trades": 787,
            "change_5d_pct_avg": 1.215,
            "change_10d_pct_avg": 1.647,
            "max_gain_20d_pct_avg": 9.956,
        },
    },
    {
        "name": "R2-confirmed",
        "touch_tolerance_of_height": 0.15,
        "min_touches": 2,
        "breakout_pct": 0.015,
        "max_height_pct": 0.20,
        "min_days_between_touches_ratio": 0.25,
        "length_days_values": (35,),
        "historical_stats": {
            "trades": 621,
            "change_5d_pct_avg": 1.167,
            "change_10d_pct_avg": 1.486,
            "max_gain_20d_pct_avg": 10.037,
        },
    },
    {
        "name": "R3-strict",
        "touch_tolerance_of_height": 0.15,
        "min_touches": 2,
        "breakout_pct": 0.02,
        "max_height_pct": 0.15,
        "min_days_between_touches_ratio": 0.25,
        "length_days_values": (65,),
        "historical_stats": {
            "trades": 125,
            "change_5d_pct_avg": 1.266,
            "change_10d_pct_avg": 1.212,
            "max_gain_20d_pct_avg": 9.691,
        },
    },
)


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


def _select_spaced_touch_clusters(mask: pd.Series, min_gap: int) -> List[tuple[int, int]]:
    clusters = _build_touch_clusters(mask)
    if not clusters:
        return []

    selected_clusters: List[tuple[int, int]] = [clusters[0]]
    last_cluster_end = clusters[0][1]

    for cluster_start, cluster_end in clusters[1:]:
        if (cluster_start - last_cluster_end) >= min_gap:
            selected_clusters.append((cluster_start, cluster_end))
            last_cluster_end = cluster_end

    return selected_clusters


def _clusters_time_span_ratio(clusters: List[tuple[int, int]], window_length: int) -> float:
    if len(clusters) < 2 or window_length <= 1:
        return 0.0

    first_start = clusters[0][0]
    last_end = clusters[-1][1]
    return (last_end - first_start) / float(window_length - 1)


def _clusters_cover_multiple_segments(clusters: List[tuple[int, int]], window_length: int) -> bool:
    if not clusters or window_length <= 0:
        return False

    segment_ids = set()
    for cluster_start, cluster_end in clusters:
        cluster_mid = (cluster_start + cluster_end) / 2.0
        normalized = cluster_mid / float(window_length)
        if normalized < (1.0 / 3.0):
            segment_ids.add(0)
        elif normalized < (2.0 / 3.0):
            segment_ids.add(1)
        else:
            segment_ids.add(2)

    return len(segment_ids) >= 2


def _has_touch_before_last_segment(clusters: List[tuple[int, int]], window_length: int) -> bool:
    if not clusters or window_length <= 0:
        return False

    cutoff = int(window_length * 0.8)
    return any(cluster_start < cutoff for cluster_start, _ in clusters)


def _is_touch_structure_valid(
        clusters: List[tuple[int, int]],
        window_length: int,
        min_touches: int,
        min_span_ratio: float,
) -> bool:
    if len(clusters) < min_touches:
        return False

    if min_touches >= 2 and len(clusters) >= 2:
        span_ratio = _clusters_time_span_ratio(clusters, window_length)
        if span_ratio < min_span_ratio:
            return False

    if not _clusters_cover_multiple_segments(clusters, window_length):
        return False

    if not _has_touch_before_last_segment(clusters, window_length):
        return False

    return True


def _format_config_label(config: dict) -> str:
    return f"📐 {config['name']}"


def _format_historical_stats(config: dict) -> str:
    stats = config["historical_stats"]
    return (
        f"🧾{stats['trades']} "
        f"5d={stats['change_5d_pct_avg']:.3f}% "
        f"10d={stats['change_10d_pct_avg']:.3f}% "
        f"20d={stats['max_gain_20d_pct_avg']:.3f}%"
    )


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

    min_touch_span_ratio = 0.40

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

        support_clusters = _select_spaced_touch_clusters(support_mask, min_days_between_touches)
        resistance_clusters = _select_spaced_touch_clusters(resistance_mask, min_days_between_touches)

        if not _is_touch_structure_valid(
                clusters=support_clusters,
                window_length=length_days,
                min_touches=min_touches,
                min_span_ratio=min_touch_span_ratio,
        ):
            continue

        if not _is_touch_structure_valid(
                clusters=resistance_clusters,
                window_length=length_days,
                min_touches=min_touches,
                min_span_ratio=min_touch_span_ratio,
        ):
            continue

        breakout_up_level = resistance * (1.0 + breakout_pct)
        breakout_down_level = support * (1.0 - breakout_pct)

        match = {
            "length_days": length_days,
            "support": support,
            "resistance": resistance,
            "height_pct": height_pct,
        }

        if close_today > breakout_up_level:
            up_matches.append(match)
        elif INCLUDE_RECTANGLE_BREAKOUT_DOWN and close_today < breakout_down_level:
            down_matches.append(match)

    parts: List[str] = []

    if up_matches:
        representative = max(up_matches, key=lambda x: x["length_days"])
        parts.append(
            f"▬ ⬆️ "
            f"{representative['length_days']}d "
            f"{representative['support']:.2f}↔{representative['resistance']:.2f}"
        )

    if down_matches:
        representative = max(down_matches, key=lambda x: x["length_days"])
        parts.append(
            f"▬ ⬇️ "
            f"{representative['length_days']}d "
            f"{representative['support']:.2f}↔{representative['resistance']:.2f}"
        )

    return " | ".join(parts) if parts else None


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


def check_rectangle_breakout_today_daily_scan(
        prices: Dict[pd.Timestamp, Dict[str, float]],
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()

    matched_parts: List[str] = []

    for config in RECTANGLE_DAILY_SCAN_CONFIGS:
        signal = _check_rectangle_breakout_on_df(
            df=df,
            touch_tolerance_of_height=config["touch_tolerance_of_height"],
            min_touches=config["min_touches"],
            breakout_pct=config["breakout_pct"],
            max_height_pct=config["max_height_pct"],
            min_days_between_touches_ratio=config["min_days_between_touches_ratio"],
            length_days_values=config["length_days_values"],
        )
        if not signal:
            continue

        config_label = _format_config_label(config)
        historical_stats = _format_historical_stats(config)
        matched_parts.append(f"{signal} [{config_label}]\n{historical_stats}")

    return "\n".join(matched_parts) if matched_parts else None