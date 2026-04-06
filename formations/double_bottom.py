from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class _Pivot:
    idx: int
    ts: pd.Timestamp
    price: float


def _find_pivot_lows(df: pd.DataFrame, left: int, right: int) -> List[_Pivot]:
    """
    Pivot low: Low w dniu i jest najniższe w oknie [i-left, i+right].
    """
    lows = df["Low"].astype(float).values
    pivots: List[_Pivot] = []
    n = len(df)

    for i in range(left, n - right):
        window = lows[i - left:i + right + 1]
        cur = lows[i]
        if cur == window.min():
            pivots.append(_Pivot(i, df.index[i], float(cur)))

    return pivots


def _max_high_between(df: pd.DataFrame, i: int, j: int) -> Tuple[int, float]:
    """
    Zwraca (idx, high) maksymalnego High w przedziale (i, j) (bez końców).
    """
    if j <= i + 1:
        return i, float(df.iloc[i]["High"])

    segment = df.iloc[i + 1:j]
    highs = segment["High"].astype(float)
    k = int(highs.values.argmax())
    idx = (i + 1) + k
    return idx, float(highs.iloc[k])


def _atr_approx(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Przybliżony ATR (Wilder-style nie jest konieczny do filtrów; wystarczy SMA TR).
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr = pd.concat(
        [
            tr,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def find_double_bottoms(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        company_abbr: Optional[str] = None,
        pivot_left: int = 3,
        pivot_right: int = 3,
        min_days_between_bottoms: int = 10,
        max_days_between_bottoms: int = 120,
        max_bottom_price_diff: float = 0.03,
        min_neckline_rise: float = 0.06,
        min_neckline_rise_from_higher_bottom: float = 0.04,
        breakout_buffer_atr: float = 0.25,
        atr_period: int = 14,
        require_downtrend_before_l1: bool = True,
        downtrend_ma_period: int = 50,
        require_drop_into_l1: bool = True,
        drop_into_l1_lookback_days: int = 30,
        min_drop_into_l1: float = 0.05,
        max_l1_close_vs_recent_high: float = 0.97,
        max_l1_close_vs_recent_avg: float = 0.98,
        max_breakout_days_after_l2: int = 60,
        breakout_days_after_l2_ratio: float = 0.6,
        breakout_days_after_l2_min: int = 5,
        breakout_days_after_l2_max: int = 40,
        forbid_close_below_bottoms_between: bool = True,
        forbid_close_below_bottoms_tolerance: float = 0.0,
        forbid_close_near_bottoms_between: bool = True,
        forbid_close_near_bottoms_max_above: float = 0.01,
        near_bottoms_exclude_days_after_l1: int = 3,
        near_bottoms_exclude_days_before_l2: int = 3,
        forbid_low_below_bottoms_between: bool = True,
        forbid_low_below_bottoms_tolerance: float = 0.0,
        low_below_exclude_days_after_l1: int = 3,
        low_below_exclude_days_before_l2: int = 3,
        forbid_any_pivot_low_between: bool = True,
        pivot_low_between_exclude_days_after_l1: int = 3,
        pivot_low_between_exclude_days_before_l2: int = 3,
        forbid_new_min_after_l2_before_breakout: bool = True,
        new_min_after_l2_exclude_days: int = 0,
        new_min_after_l2_tolerance: float = 0.0,
        debug_print: bool = False,
) -> List[dict]:
    """
    Znajduje historyczne formacje podwójnego dna (W) w dostarczonych danych.

    Filtry jakości:
    1) forbid_close_below_bottoms_between:
       pomiędzy L1 i L2 (bez L1/L2) NIE MOŻE BYĆ zamknięcia (Close) poniżej min(L1,L2)
       (z ewentualną tolerancją w dół).

    2) forbid_close_near_bottoms_between:
       pomiędzy L1 i L2 (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być Close "na dołku",
       tj. <= min(L1,L2) * (1 + max_above). Domyślnie max_above=1%.

    3) forbid_low_below_bottoms_between:
       pomiędzy L1 i L2 (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być żadnego
       intraday Low niższego od min(L1_low, L2_low) (z ewentualną tolerancją w dół).

    4) forbid_any_pivot_low_between:
       pomiędzy L1 i L2 (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być żadnego
       dodatkowego pivot-lowa. To eliminuje przypadki "triple bottom / multiple bottoms",
       nawet jeśli ten dodatkowy dołek nie robi nowego minimum.

    5) require_drop_into_l1:
       przed L1 musi być widoczny spadek (żeby nie łapać L1 jako części ruchu horyzontalnego).
       Realizacja: w oknie lookback przed L1 wymagamy:
         - spadku L1-close od lokalnego high o co najmniej min_drop_into_l1,
         - oraz L1-close istotnie poniżej lokalnego high i średniej close w tym oknie.

    Uwaga: (1) i (2) działają na Close, (3) na Low, (4) na pivotach.
    """
    if not prices:
        return []

    df = pd.DataFrame.from_dict(prices, orient="index").dropna().sort_index()

    required_cols = {"Close", "High", "Low"}
    if not required_cols.issubset(set(df.columns)):
        return []

    df = df[["Close", "High", "Low"]].copy()
    if len(df) < (pivot_left + pivot_right + 1) + 30:
        return []

    atr = _atr_approx(df, atr_period)
    pivots = _find_pivot_lows(df, pivot_left, pivot_right)
    if len(pivots) < 2:
        return []

    ma = None
    if require_downtrend_before_l1:
        ma = df["Close"].astype(float).rolling(
            downtrend_ma_period,
            min_periods=max(5, downtrend_ma_period // 2)
        ).mean()

    results: List[dict] = []
    n = len(df)

    closes = df["Close"].astype(float).values
    lows = df["Low"].astype(float).values

    for a in range(len(pivots)):
        l1 = pivots[a]

        # 5) L1 musi wynikać ze spadku (anty-horyzontal)
        if require_drop_into_l1:
            lb = max(5, int(drop_into_l1_lookback_days))
            start_idx = max(0, l1.idx - lb)
            if start_idx >= l1.idx:
                continue

            recent = df.iloc[start_idx:l1.idx]
            if len(recent) < 5:
                continue

            recent_high = float(recent["High"].astype(float).max())
            recent_close_avg = float(recent["Close"].astype(float).mean())
            l1_close = float(df.iloc[l1.idx]["Close"])

            if recent_high > 0:
                drop_from_recent_high = (recent_high - l1_close) / recent_high
                if drop_from_recent_high < min_drop_into_l1:
                    continue

            if l1_close > recent_high * float(max_l1_close_vs_recent_high):
                continue
            if l1_close > recent_close_avg * float(max_l1_close_vs_recent_avg):
                continue

        # Opcjonalny filtr trendu przed L1
        if require_downtrend_before_l1 and ma is not None:
            ma_l1 = ma.iloc[l1.idx]
            if pd.notna(ma_l1):
                if float(df.iloc[l1.idx]["Close"]) >= float(ma_l1):
                    continue

        for b in range(a + 1, len(pivots)):
            l2 = pivots[b]

            days_between = l2.idx - l1.idx
            if days_between < min_days_between_bottoms:
                continue
            if days_between > max_days_between_bottoms:
                break

            bottom_min = min(l1.price, l2.price)
            bottom_max = max(l1.price, l2.price)
            if bottom_min <= 0:
                continue

            # Różnica dołków jako % niższego dołka; default 2% (podwójne dno ma być "równe")
            bottom_diff = (bottom_max - bottom_min) / bottom_min
            if bottom_diff > max_bottom_price_diff:
                continue

            # 1) Między dołkami nie może być zamknięcia poniżej dołków (Close)
            if forbid_close_below_bottoms_between and (l2.idx > l1.idx + 1):
                threshold = bottom_min * (1.0 - forbid_close_below_bottoms_tolerance)
                between_closes = closes[l1.idx + 1:l2.idx]
                if len(between_closes) > 0 and float(between_closes.min()) < float(threshold):
                    continue

            # 2) Między dołkami (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być Close "na dołku"
            if forbid_close_near_bottoms_between and (l2.idx > l1.idx + 1):
                near_threshold = bottom_min * (1.0 + forbid_close_near_bottoms_max_above)

                start = l1.idx + 1 + max(0, near_bottoms_exclude_days_after_l1)
                end = l2.idx - max(0, near_bottoms_exclude_days_before_l2)

                if start < end:
                    mid_closes = closes[start:end]
                    if len(mid_closes) > 0 and float(mid_closes.min()) <= float(near_threshold):
                        continue

            # 3) Między dołkami (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być Low niżej niż dołki
            if forbid_low_below_bottoms_between and (l2.idx > l1.idx + 1):
                low_threshold = bottom_min * (1.0 - forbid_low_below_bottoms_tolerance)

                start = l1.idx + 1 + max(0, low_below_exclude_days_after_l1)
                end = l2.idx - max(0, low_below_exclude_days_before_l2)

                if start < end:
                    mid_lows = lows[start:end]
                    if len(mid_lows) > 0 and float(mid_lows.min()) < float(low_threshold):
                        continue

            # 4) Między dołkami (z wyjątkiem X dni po L1 i Y dni przed L2) nie może być dodatkowego pivot-lowa
            if forbid_any_pivot_low_between:
                start = l1.idx + 1 + max(0, pivot_low_between_exclude_days_after_l1)
                end = l2.idx - max(0, pivot_low_between_exclude_days_before_l2)
                if start < end:
                    has_pivot_between = any((p.idx >= start) and (p.idx <= end) for p in pivots[a + 1:b])
                    if has_pivot_between:
                        continue

            neckline_idx, neckline = _max_high_between(df, l1.idx, l2.idx)
            if neckline <= 0:
                continue

            rise = (neckline - bottom_min) / bottom_min
            if rise < min_neckline_rise:
                continue

            # szyja musi być co najmniej X% powyżej WYŻSZEGO dołka (żeby "W" nie było zbyt płaskie).
            higher_bottom = bottom_max
            rise_from_higher = (neckline - higher_bottom) / higher_bottom
            if rise_from_higher < min_neckline_rise_from_higher_bottom:
                continue

            # --- breakout window: proporcjonalne do wielkości formacji ---
            allowed_days_after_l2 = int(round(days_between * breakout_days_after_l2_ratio))
            allowed_days_after_l2 = max(breakout_days_after_l2_min, allowed_days_after_l2)
            allowed_days_after_l2 = min(breakout_days_after_l2_max, allowed_days_after_l2)
            allowed_days_after_l2 = min(max_breakout_days_after_l2, allowed_days_after_l2)

            # Szukamy pierwszego breakout po L2
            breakout_idx: Optional[int] = None
            breakout_close: Optional[float] = None

            search_start = l2.idx + 1
            search_end = min(n - 1, l2.idx + allowed_days_after_l2)

            if search_start > search_end:
                continue

            for k in range(search_start, search_end + 1):
                close_k = float(df.iloc[k]["Close"])
                atr_k = float(atr.iloc[k]) if pd.notna(atr.iloc[k]) else 0.0
                buffer = (breakout_buffer_atr * atr_k) if atr_k > 0 else 0.0

                if close_k > neckline + buffer:
                    breakout_idx = k
                    breakout_close = close_k
                    break

            if breakout_idx is None or breakout_close is None:
                continue

            # --- po L2 a przed breakout NIE może być nowego minimum (intraday Low) ---
            if forbid_new_min_after_l2_before_breakout:
                start_nm = l2.idx + 1 + max(0, new_min_after_l2_exclude_days)
                end_nm = breakout_idx  # bez dnia breakout
                if start_nm < end_nm:
                    post_l2_min_low = float(lows[start_nm:end_nm].min())
                    threshold = bottom_min * (1.0 - new_min_after_l2_tolerance)
                    if post_l2_min_low < threshold:
                        continue

            # Prosty score: większe rise, mniejsza różnica dołków; kara za zbyt długi układ i późny breakout.
            length_penalty = (days_between / max_days_between_bottoms)
            breakout_delay = breakout_idx - l2.idx
            breakout_penalty = (breakout_delay / max(1, allowed_days_after_l2))
            score = rise * (1.0 - bottom_diff) * (1.0 - 0.5 * length_penalty) * (1.0 - 0.5 * breakout_penalty)

            target = neckline + (neckline - bottom_min)

            result = {
                "company_abbr": company_abbr,
                "l1_idx": l1.idx,
                "l2_idx": l2.idx,
                "neckline_idx": neckline_idx,
                "breakout_idx": breakout_idx,
                "l1_date": l1.ts,
                "l2_date": l2.ts,
                "neckline_date": df.index[neckline_idx],
                "breakout_date": df.index[breakout_idx],
                "l1_price": float(l1.price),
                "l2_price": float(l2.price),
                "neckline": float(neckline),
                "breakout_close": float(breakout_close),
                "bottom_diff": float(bottom_diff),
                "rise": float(rise),
                "target": float(target),
                "score": float(score),
            }

            if debug_print:
                prefix = f"[{company_abbr}] " if company_abbr else ""
                print(f"{prefix}PODWOJNE DNO:")
                print(result)
                print("\n" * 3)

            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def format_double_bottoms(double_bottoms: List[dict], limit: int = 10) -> str:
    """
    Pomocniczo: formatuje listę wyników do czytelnego tekstu (np. do raportu / printa).
    """
    if not double_bottoms:
        return ""

    lines: List[str] = []
    for i, r in enumerate(double_bottoms[:limit], start=1):
        company = r.get("company_abbr") or ""
        company_prefix = f"{company} " if company else ""

        l1 = r["l1_date"].strftime("%Y-%m-%d")
        l2 = r["l2_date"].strftime("%Y-%m-%d")
        nk = r["neckline_date"].strftime("%Y-%m-%d")
        bo = r["breakout_date"].strftime("%Y-%m-%d")

        print(
            company + " " + l1 + "(" + str(r["l1_price"]) + ") / " + l2 + "(" + str(r["l2_price"]) + ")"
            + " | neck " + nk + " | breakout " + bo + " | target~" + str(r["target"])
        )

        lines.append(
            f"{i:02d}) {company_prefix}W {l1} ({r['l1_price']:.2f}) / {l2} ({r['l2_price']:.2f}) | "
            f"neck {nk}={r['neckline']:.2f} | breakout {bo} close={r['breakout_close']:.2f} | "
            f"target~{r['target']:.2f} | score={r['score']:.4f}"
        )

    return "\n".join(lines)


def check_double_bottom_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        company_abbr: Optional[str] = None,
        **kwargs,
) -> Optional[str]:
    """
    Kompatybilność: zwraca sygnał tylko dla wybicia DZISIAJ.

    Implementacja: znajdujemy wszystkie historyczne formacje, a potem wybieramy tę,
    której breakout_date == ostatni dzień w danych (dzisiaj), o najwyższym score.
    """
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").dropna().sort_index()
    if df.empty:
        return None
    today_ts = df.index[-1]

    all_found = find_double_bottoms(prices, company_abbr=company_abbr, **kwargs)
    todays = [r for r in all_found if r.get("breakout_date") == today_ts]
    if not todays:
        return None

    best = max(todays, key=lambda x: x["score"])
    l1 = best["l1_date"].strftime("%m-%d")
    l2 = best["l2_date"].strftime("%m-%d")
    neckline = float(best["neckline"])
    target = float(best["target"])

    result = f"🆆{l1}/{l2} (neck={neckline:.2f}, target~{target:.2f})"
    print(result)
    return result