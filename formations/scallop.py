"""
Detekcja formacji odwróconej muszli zwyżkującej (Inverted Ascending Scallop)
wg Thomasa Bulkowskiego.

Kształt: odwrócona litera J
- Punkt A: początek formacji — cena zaczyna rosnąć (liniowo lub po łuku)
- Punkt C: wierzchołek — zaokrąglony szczyt
- Punkt B: dno korekty po wierzchołku
- Wybicie: Close > C (powyżej wierzchołka formacji)

Reguły identyfikacji (Bulkowski):
1. Trend wzrostowy od A do C — cena rośnie po linii prostej lub łagodnym łuku
2. Wierzchołek C zaokrąglony (zwrot stopniowy)
3. Korekta B: zniesienie ruchu A→C średnio 53%, max 62%
4. Wybicie potwierdzone przez Close > C
5. Mediana czasu od B do wybicia: 17 dni
6. Czas wybicia ≤ 17 dni → avg zysk 44%, > 17 dni → 34%
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Close", "High", "Low"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna(subset=["Close", "High", "Low"]).sort_index()


def _find_local_maxima(df: pd.DataFrame, order: int = 5) -> List[int]:
    """Lokalne maksima High z marginesem `order` świec po każdej stronie."""
    highs = df["High"].values
    n = len(highs)
    maxima: List[int] = []
    for i in range(order, n - order):
        window = highs[i - order: i + order + 1]
        if highs[i] == window.max() and highs[i] > window.mean():
            maxima.append(i)
    return maxima


def _find_local_minima(df: pd.DataFrame, order: int = 5) -> List[int]:
    """Lokalne minima Low z marginesem `order` świec po każdej stronie."""
    lows = df["Low"].values
    n = len(lows)
    minima: List[int] = []
    for i in range(order, n - order):
        window = lows[i - order: i + order + 1]
        if lows[i] == window.min():
            minima.append(i)
    return minima


def _arc_smoothness(df: pd.DataFrame, start_idx: int, peak_idx: int, end_idx: int) -> float:
    """
    Mierzy gładkość łuku A→C→B jako R² dopasowania parabolicznego do High.
    Wartość bliska 1.0 = gładki łuk (kształt muszli).
    Wartość bliska 0.0 = chaotyczny/skokowy ruch.

    Używa High całego okna A→B żeby ocenić czy kształt przypomina odwróconą parabolę.
    """
    segment = df.iloc[start_idx: end_idx + 1]["High"].values
    n = len(segment)
    if n < 5:
        return 0.0

    x = np.arange(n, dtype=float)
    # Dopasowanie kwadratowe (parabola)
    try:
        coeffs = np.polyfit(x, segment, 2)
        fitted = np.polyval(coeffs, x)
    except Exception:
        return 0.0

    # R² = 1 - SS_res/SS_tot
    ss_res = float(np.sum((segment - fitted) ** 2))
    ss_tot = float(np.sum((segment - segment.mean()) ** 2))
    if ss_tot == 0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, r2))


def _is_rising_trend(df: pd.DataFrame, start_idx: int, end_idx: int,
                     min_rise_pct: float = 0.10) -> bool:
    """
    Sprawdź czy od start_idx do end_idx cena rośnie (trend wzrostowy A→C).
    Wymaga:
    - wzrost Close ≥ min_rise_pct
    - co najmniej połowa okna ma rosnący Close (ogólny kierunek wzrostowy)
    """
    if end_idx <= start_idx + 2:
        return False
    segment = df.iloc[start_idx: end_idx + 1]["Close"].values
    if len(segment) < 3:
        return False
    start_close = float(segment[0])
    end_close = float(segment[-1])
    if start_close <= 0:
        return False
    rise = (end_close - start_close) / start_close
    if rise < min_rise_pct:
        return False
    # Regresja liniowa — nachylenie musi być dodatnie
    x = np.arange(len(segment))
    slope = float(np.polyfit(x, segment, 1)[0])
    return slope > 0


def _is_rounded_top(df: pd.DataFrame, peak_idx: int, window: int = 5) -> bool:
    """
    Sprawdź czy wierzchołek jest zaokrąglony.
    Zaokrąglony szczyt: High w okolicach szczytu nie spada gwałtownie —
    sąsiednie High są bliskie szczytowi (różnica < 5% od peak High).
    """
    n = len(df)
    lo = max(0, peak_idx - window)
    hi = min(n - 1, peak_idx + window)
    peak_high = float(df.iloc[peak_idx]["High"])
    nearby = df.iloc[lo: hi + 1]["High"].values
    # Zaokrąglony = co najmniej 40% sąsiednich świec ma High w obrębie 8% szczytu
    threshold = peak_high * 0.92
    near_count = int((nearby >= threshold).sum())
    return near_count >= max(2, int(0.4 * len(nearby)))


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def find_scallop_signals(
        df: pd.DataFrame,
        # --- parametry formacji ---
        local_order: int = 7,               # okno dla lokalnych ekst.
        min_ac_rise_pct: float = 0.25,      # min wzrost A→C
        min_ac_days: int = 15,              # min długość ruchu A→C
        max_ac_days: int = 90,              # max długość ruchu A→C
        min_retracement: float = 0.40,      # min zniesienie — Bulkowski avg 54%
        max_retracement: float = 0.90,      # max zniesienie — Bulkowski: unikaj 100%
        max_breakout_days: int = 30,        # Bulkowski mediana 17 dni — max ~1.75× mediany
        # --- gładkość łuku ---
        min_arc_smoothness: float = 0.90,   # R² ≥ 0.90 — bardzo gładki łuk muszli
        # --- czystość wzrostu A→C (Bulkowski: "nearly straight run up") ---
        max_rise_throwback: float = 0.12,   # max 12% cofnięcie od bieżącego szczytu w trakcie wzrostu
        # --- uptrend przed A ---
        require_uptrend_before_a: bool = True,
        uptrend_lookback: int = 40,
        min_uptrend_pct: float = 0.15,
        # --- wolumen (Bulkowski: maleje w 70% formacji) ---
        check_volume_decline: bool = False,
) -> List[dict]:
    """
    Skanuj DataFrame i zwróć listę potwierdzonych formacji odwróconej muszli zwyżkującej.

    Każdy rekord zawiera:
      date             – data wybicia (Close > C)
      signal           – opis tekstowy
      a_date / a_price – początek muszli
      c_date / c_price – wierzchołek (peak)
      b_date / b_price – dno korekty
      breakout_price   – cena wybicia
      retracement_pct  – zniesienie C→B jako % ruchu A→C
      ac_rise_pct      – wzrost A→C
      ac_days          – długość ruchu A→C
      bc_days          – długość korekty B
      breakout_days    – dni od B do wybicia
    """
    wdf = _prepare_df(df)
    if wdf.empty or len(wdf) < local_order * 2 + min_ac_days + 5:
        return []

    maxima = _find_local_maxima(wdf, order=local_order)
    minima = _find_local_minima(wdf, order=local_order)
    n = len(wdf)
    results: List[dict] = []
    confirmed_breakout_idx = -1

    for ci, c_idx in enumerate(maxima):
        if c_idx <= confirmed_breakout_idx:
            continue

        c_high = float(wdf.iloc[c_idx]["High"])

        # Znajdź punkt A: lokalne minimum przed C (start rosnącego ramienia)
        # A musi być co najmniej min_ac_days przed C i co najwyżej max_ac_days
        candidates_a = [
            m for m in minima
            if min_ac_days <= (c_idx - m) <= max_ac_days
        ]
        if not candidates_a:
            continue

        # Wybierz A jako najwcześniejsze minimum które daje najczystszy wzrost
        for a_idx in candidates_a:
            a_low = float(wdf.iloc[a_idx]["Low"])
            if a_low <= 0:
                continue

            ac_rise = (c_high - a_low) / a_low
            if ac_rise < min_ac_rise_pct:
                continue

            # Sprawdź trend wzrostowy A→C
            if not _is_rising_trend(wdf, a_idx, c_idx, min_rise_pct=min_ac_rise_pct * 0.7):
                continue

            # ── Filtr: silny wzrost na początku (Bulkowski: "nearly straight run up") ──
            # Pierwsza połowa ruchu A→C musi zawierać co najmniej 55% całego wzrostu.
            # Formacja muszli zaczyna się od dynamicznego impulsu, który zwalnia przy szczycie.
            mid_idx = a_idx + (c_idx - a_idx) // 2
            mid_close = float(wdf.iloc[mid_idx]["Close"])
            a_close = float(wdf.iloc[a_idx]["Close"])
            c_close = float(wdf.iloc[c_idx]["Close"])
            total_rise = c_close - a_close
            first_half_rise = mid_close - a_close
            if total_rise > 0 and (first_half_rise / total_rise) < 0.45:
                # Wzrost skupiony w drugiej połowie — to nie jest muszla, to ruch liniowy lub odwrócony
                continue

            # ── Filtr: czysty wzrost A→C bez istotnych cofnięć ──
            # Bulkowski: "nearly straight run up" — max cofnięcie od bieżącego szczytu
            # w trakcie wzrostu nie może przekroczyć max_rise_throwback.
            # Analogia do max_throwback_in_decline z double_bottom.
            closes_ac = wdf["Close"].values[a_idx: c_idx + 1]
            running_max = closes_ac[0]
            max_throwback_rise = 0.0
            for c in closes_ac[1:]:
                if c > running_max:
                    running_max = c
                elif running_max > 0:
                    throwback = (running_max - c) / running_max
                    if throwback > max_throwback_rise:
                        max_throwback_rise = throwback
            if max_throwback_rise > max_rise_throwback:
                continue

            # Sprawdź zaokrąglony wierzchołek C
            if not _is_rounded_top(wdf, c_idx, window=local_order):
                continue

            # Sprawdź gładkość łuku (Bulkowski: "nearly straight run up, rounded at top")
            # Mierzymy R² parabolicznego dopasowania do całego okna A→C
            # Wymagamy umiarkowanie gładkiego łuku żeby odrzucić skokowe/rwane formacje
            if min_arc_smoothness > 0:
                smoothness = _arc_smoothness(wdf, a_idx, c_idx, c_idx)
                if smoothness < min_arc_smoothness:
                    continue

            # Sprawdź trend wzrostowy przed A (opcjonalnie)
            if require_uptrend_before_a:
                lb_start = max(0, a_idx - uptrend_lookback)
                pre_a = wdf.iloc[lb_start: a_idx + 1]["Close"]
                if len(pre_a) >= 5:
                    pre_rise = (float(pre_a.iloc[-1]) - float(pre_a.iloc[0])) / float(pre_a.iloc[0])
                    if pre_rise < min_uptrend_pct:
                        continue

            # Znajdź B: lokalne minimum po C (dno korekty)
            candidates_b = [
                m for m in minima
                if m > c_idx and (m - c_idx) <= max_breakout_days + 10
            ]
            if not candidates_b:
                continue

            b_idx = candidates_b[0]  # pierwsze minimum po C
            b_low = float(wdf.iloc[b_idx]["Low"])

            # Zniesienie C→B jako % ruchu A→C
            ac_height = c_high - a_low
            if ac_height <= 0:
                continue
            retracement = (c_high - b_low) / ac_height

            if retracement < min_retracement or retracement > max_retracement:
                continue

            # B nie może być niżej niż A (formacja musi być zwyżkująca)
            if b_low < a_low:
                continue

            # Szukaj wybicia: Close > c_high po B
            search_start = b_idx + 1
            search_end = min(n, b_idx + max_breakout_days + 1)

            breakout_idx: Optional[int] = None
            for bi in range(search_start, search_end):
                close_bi = float(wdf.iloc[bi]["Close"])
                # Unieważnienie: Close < b_low
                if close_bi < b_low:
                    break
                if close_bi > c_high:
                    breakout_idx = bi
                    break

            if breakout_idx is None:
                continue

            # Buduj sygnał
            a_date = wdf.index[a_idx]
            c_date = wdf.index[c_idx]
            b_date = wdf.index[b_idx]
            breakout_date = wdf.index[breakout_idx]
            breakout_price = float(wdf.iloc[breakout_idx]["Close"])
            ac_days = c_idx - a_idx
            bc_days = b_idx - c_idx
            breakout_days = breakout_idx - b_idx

            signal = (
                f"🐚{a_date.strftime('%Y-%m-%d')}→{c_date.strftime('%Y-%m-%d')}→{b_date.strftime('%Y-%m-%d')} "
                f"(conf={c_high:.2f}, ret={retracement*100:.0f}%)"
            )

            results.append({
                "date": breakout_date,
                "signal": signal,
                "a_date": a_date,
                "c_date": c_date,
                "b_date": b_date,
                "a_price": a_low,
                "c_price": c_high,
                "b_price": b_low,
                "breakout_price": breakout_price,
                "retracement_pct": round(retracement * 100, 2),
                "ac_rise_pct": round(ac_rise * 100, 2),
                "ac_days": ac_days,
                "bc_days": bc_days,
                "breakout_days": breakout_days,
                "a_idx": a_idx,
                "c_idx": c_idx,
                "b_idx": b_idx,
                "breakout_idx": breakout_idx,
            })

            confirmed_breakout_idx = breakout_idx
            break  # jeden C per iteracja

    return results


# ---------------------------------------------------------------------------
# Single-bar check
# ---------------------------------------------------------------------------

def _check_scallop_on_df(
        df: pd.DataFrame,
        local_order: int = 7,
        min_ac_rise_pct: float = 0.25,
        min_ac_days: int = 15,
        max_ac_days: int = 90,
        min_retracement: float = 0.40,
        max_retracement: float = 0.90,
        max_breakout_days: int = 30,
        min_arc_smoothness: float = 0.85,
        max_rise_throwback: float = 0.08,
        require_uptrend_before_a: bool = True,
        uptrend_lookback: int = 40,
        min_uptrend_pct: float = 0.15,
        check_volume_decline: bool = False,
) -> Optional[str]:
    signals = find_scallop_signals(
        df=df,
        local_order=local_order,
        min_ac_rise_pct=min_ac_rise_pct,
        min_ac_days=min_ac_days,
        max_ac_days=max_ac_days,
        min_retracement=min_retracement,
        max_retracement=max_retracement,
        max_breakout_days=max_breakout_days,
        min_arc_smoothness=min_arc_smoothness,
        max_rise_throwback=max_rise_throwback,
        require_uptrend_before_a=require_uptrend_before_a,
        uptrend_lookback=uptrend_lookback,
        min_uptrend_pct=min_uptrend_pct,
        check_volume_decline=check_volume_decline,
    )
    if not signals:
        return None
    last = signals[-1]
    if pd.Timestamp(last["date"]) != df.index[-1]:
        return None
    return str(last["signal"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_scallop_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        local_order: int = 7,
        min_ac_rise_pct: float = 0.25,
        min_ac_days: int = 15,
        max_ac_days: int = 90,
        min_retracement: float = 0.40,
        max_retracement: float = 0.90,
        max_breakout_days: int = 30,
        min_arc_smoothness: float = 0.85,
        max_rise_throwback: float = 0.08,
        require_uptrend_before_a: bool = True,
        uptrend_lookback: int = 40,
        min_uptrend_pct: float = 0.15,
        check_volume_decline: bool = False,
) -> Optional[str]:
    """
    Sprawdź czy dzisiaj (ostatni bar) pojawia się wybicie z formacji muszli.
    `prices` – słownik {timestamp: {Open, High, Low, Close, Volume}}.
    """
    if not prices:
        return None
    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_scallop_on_df(
        df=df,
        local_order=local_order,
        min_ac_rise_pct=min_ac_rise_pct,
        min_ac_days=min_ac_days,
        max_ac_days=max_ac_days,
        min_retracement=min_retracement,
        max_retracement=max_retracement,
        max_breakout_days=max_breakout_days,
        min_arc_smoothness=min_arc_smoothness,
        max_rise_throwback=max_rise_throwback,
        require_uptrend_before_a=require_uptrend_before_a,
        uptrend_lookback=uptrend_lookback,
        min_uptrend_pct=min_uptrend_pct,
        check_volume_decline=check_volume_decline,
    )
