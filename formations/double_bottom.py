"""
Detekcja formacji podwójnego dna (Eve & Eve / Adam & Adam / Eve & Adam / Adam & Eve)
wg Thomasa Bulkowskiego (Encyclopedia of Chart Patterns, Swing and Day Trading).

Reguły identyfikacji (Eve & Eve jako bazowa, wspólna dla wszystkich wariantów):
- Poprzedzający trend spadkowy
- Dwa wyraźne dołki na zbliżonym poziomie cenowym (różnica ≤ max_bottom_diff_pct, domyślnie 6%)
- Wzrost między dołkami ≥ min_peak_rise_pct (domyślnie 10%)
- Odstęp między dołkami: min_separation_days … max_separation_days (domyślnie 2–20 tygodni)
- Wariant dna:
    Adam = wąski, spiczasty dołek (mało świec w obrębie minimum, jeden dominujący szpikułec)
    Eve  = szeroki, zaokrąglony dołek (dużo świec na podobnym poziomie, krótkie szpikułce)
- Potwierdzenie = zamknięcie powyżej szczytu między dołkami (linia potwierdzenia)
- Wolumen zazwyczaj wyższy na lewym dnie (sprawdzany opcjonalnie)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def _classify_bottom(df: pd.DataFrame, trough_idx: int, window: int = 5) -> str:
    """
    Klasyfikuj dno jako 'Adam' lub 'Eve'.

    Adam: wąski, spiczasty – mało świec wewnątrz okna leży blisko minimum,
          dominuje jeden długi cień dolny (szpikułec).
    Eve:  szeroki, zaokrąglony – dużo świec blisko minimum, cienie krótkie.

    Parametr `window` to połowa szerokości okna po obu stronach trough_idx.
    """
    n = len(df)
    lo = max(0, trough_idx - window)
    hi = min(n - 1, trough_idx + window)
    slice_df = df.iloc[lo: hi + 1]

    trough_low = float(df.iloc[trough_idx]["Low"])
    if trough_low <= 0:
        return "Eve"

    # Ile świec ma Low w obrębie 3% minimum?
    proximity_threshold = trough_low * 1.03
    near_bottom_count = int((slice_df["Low"] <= proximity_threshold).sum())

    # Długość dolnego cienia dominującej świecy (szpikułec)
    if "Open" in df.columns:
        body_lows = df.iloc[lo: hi + 1].apply(
            lambda r: min(float(r["Open"]), float(r["Close"])), axis=1
        )
    else:
        body_lows = slice_df["Close"]

    shadow_lengths = slice_df["Low"] - body_lows
    max_shadow = float(shadow_lengths.min())  # min bo Low < body_low → wartości ujemne
    avg_shadow = float(shadow_lengths.mean())

    # Adam: wąski dołek — dominuje jeden szpikułec LUB mało świec blisko dna.
    # Warunki są rozłączne (OR), nie łączne — wystarczy spełnić jeden z nich.
    spike_ratio = max_shadow / avg_shadow if avg_shadow != 0 else 1.0
    sharp_spike = spike_ratio < 0.4          # jeden cień wyraźnie dłuższy od średniej
    narrow_base = near_bottom_count <= 3     # co najwyżej 3 świece blisko minimum w oknie ±5
    if sharp_spike or narrow_base:
        return "Adam"
    return "Eve"


def _find_local_minima(
    df: pd.DataFrame,
    order: int = 5,
    min_depth_pct: float = 0.01,  # dołek musi być co najmniej 1% niżej od średniej okna
) -> List[int]:
    """
    Znajdź lokalne minima (Low) z marginesem `order` świec po każdej stronie.

    Dwa warunki muszą być spełnione łącznie:
    1. lows[i] jest ściśle mniejsze od WSZYSTKICH sąsiadów w oknie (nie tylko <=).
       Eliminuje płaskie konsolidacje gdzie kilka dni ma identyczny Low.
    2. lows[i] jest co najmniej min_depth_pct% niżej od średniej Low w oknie.
       Wymaga wyraźnego, izolowanego dna — nie fragmentu poziomej bazy.
    """
    lows = df["Low"].values
    n = len(lows)
    minima: List[int] = []
    for i in range(order, n - order):
        window = lows[i - order: i + order + 1]
        cur = lows[i]

        # Warunek 1: ściśle mniejszy od wszystkich pozostałych w oknie
        others = [lows[j] for j in range(i - order, i + order + 1) if j != i]
        if any(cur >= other for other in others):
            continue

        # Warunek 2: wyraźnie poniżej średniej okna (prawdziwy dołek, nie konsolidacja)
        avg_window = float(sum(window) / len(window))
        if avg_window <= 0 or (avg_window - cur) / avg_window < min_depth_pct:
            continue

        minima.append(i)
    return minima


def _find_prior_downtrend(
    df: pd.DataFrame,
    left_trough_idx: int,
    lookback: int = 60,
    min_decline_pct: float = 0.10,
) -> bool:
    """
    Sprawdź czy przed lewym dnem był trend spadkowy ≥ min_decline_pct.
    Patrzymy wstecz max `lookback` świec od lewego dna.
    """
    start = max(0, left_trough_idx - lookback)
    window = df.iloc[start: left_trough_idx + 1]
    if len(window) < 5:
        return False
    peak_close = float(window["Close"].max())
    trough_low = float(df.iloc[left_trough_idx]["Low"])
    if peak_close <= 0:
        return False
    decline = (peak_close - trough_low) / peak_close
    return decline >= min_decline_pct


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def find_double_bottom_signals(
    df: pd.DataFrame,
    # --- dołki ---
    local_min_order: int = 5,           # okno dla lokalnych minimów
    min_separation_days: int = 10,      # min odstęp między dołkami (dni)
    max_separation_days: int = 70,      # max odstęp (≈14 tygodni, Bulkowski typowy zakres 2–7 tyg)
    max_bottom_diff_pct: float = 0.06,  # max różnica wysokości dołków (6%)
    # --- szczyt między dołkami ---
    min_peak_rise_pct: float = 0.17,    # min wzrost od dna do szczytu (17%, kompromis między min 10% a medianą 19%)
    # --- trend poprzedzający (długi: 60 sesji) ---
    require_downtrend: bool = True,
    downtrend_lookback: int = 60,
    min_downtrend_pct: float = 0.15,
    # --- stromy zjazd tuż przed L1 (Bulkowski: "Big W / steep decline into L1") ---
    require_drop_into_l1: bool = True,
    drop_into_l1_lookback: int = 30,    # ile sesji przed L1 sprawdzamy
    min_drop_into_l1: float = 0.08,     # min 8% spadek od lokalnego szczytu do L1-close
    # --- klasyfikacja wariantu ---
    bottom_window: int = 5,             # okno klasyfikacji Adam/Eve
    # --- wolumen ---
    check_volume: bool = False,         # Bulkowski: "usually" higher on left — obserwacja, nie twarda reguła
) -> List[dict]:
    """
    Skanuj DataFrame i zwróć listę potwierdzonych formacji podwójnego dna.

    Każdy rekord zawiera:
      date             – data potwierdzenia (zamknięcie > szczyt między dołkami)
      signal           – opis tekstowy formacji
      left_trough_date / right_trough_date
      left_trough_price / right_trough_price
      peak_date / peak_price (szczyt między dołkami = linia potwierdzenia)
      confirmation_price
      pattern_type     – 'Adam & Adam', 'Adam & Eve', 'Eve & Adam', 'Eve & Eve'
      pattern_height   – wysokość formacji (confirmation_price – niższe z dna)
      prior_downtrend  – bool
      volume_ok        – bool
    """
    wdf = _prepare_df(df)
    if wdf.empty or len(wdf) < local_min_order * 2 + min_separation_days + 1:
        return []

    minima = _find_local_minima(wdf, order=local_min_order)
    n = len(wdf)
    results: List[dict] = []

    confirmed_breakout_idx = -1  # unikaj duplikatów w tym samym oknie

    for li in range(len(minima) - 1):
        left_idx = minima[li]
        if left_idx <= confirmed_breakout_idx:
            continue

        for ri in range(li + 1, len(minima)):
            right_idx = minima[ri]
            separation = right_idx - left_idx
            if separation < min_separation_days:
                continue
            if separation > max_separation_days:
                break  # lista posortowana rosnąco

            left_low = float(wdf.iloc[left_idx]["Low"])
            right_low = float(wdf.iloc[right_idx]["Low"])

            if left_low <= 0 or right_low <= 0:
                continue

            # Różnica wysokości dołków
            bottom_diff = abs(left_low - right_low) / min(left_low, right_low)
            if bottom_diff > max_bottom_diff_pct:
                continue

            lower_trough = min(left_low, right_low)

            # ── Filtr 1: między L1 a L2 Close nie może spaść więcej niż 2% poniżej dolnego dna ──
            # Tolerancja 2% akceptuje małe wahania / szpikułce nie naruszające struktury formacji.
            between_closes = wdf["Close"].values[left_idx + 1: right_idx]
            if len(between_closes) > 0 and float(between_closes.min()) < lower_trough * 0.98:
                continue

            # ── Filtr 2: między L1 a L2 nie może być dodatkowego lokalnego minimum ──
            # Margin = local_min_order, żeby nie odrzucać małych dołków przy samych krawędziach
            # które są po prostu "echo" L1/L2 w oknie pivot detection.
            margin = local_min_order
            inner_start = li + 1
            inner_end = ri
            has_inner_pivot = any(
                (minima[k] >= left_idx + margin) and (minima[k] <= right_idx - margin)
                for k in range(inner_start, inner_end)
            )
            if has_inner_pivot:
                continue

            # Znajdź szczyt (peak) między dołkami
            between = wdf.iloc[left_idx: right_idx + 1]
            peak_label = between["High"].idxmax()
            peak_pos = wdf.index.get_loc(peak_label)  # type: ignore[arg-type]

            peak_high = float(wdf.iloc[peak_pos]["High"])

            # ── Filtr 3: peak musi leżeć w środkowej części rozpiętości L1→L2 ──
            # 0.25–0.75: środkowe 50% zakresu czasowego — luźniejsze niż wcześniej,
            # ale nadal zapobiega szczytowi przyklejonemu do samego L1 lub L2.
            span = right_idx - left_idx
            peak_pos_ratio = (peak_pos - left_idx) / span if span > 0 else 0.5
            if peak_pos_ratio < 0.25 or peak_pos_ratio > 0.75:
                continue

            # Wzrost od niższego dna do szczytu
            peak_rise = (peak_high - lower_trough) / lower_trough
            if peak_rise < min_peak_rise_pct:
                continue

            # Trend poprzedzający (ogólny, 60 sesji)
            prior_down = _find_prior_downtrend(
                wdf, left_idx,
                lookback=downtrend_lookback,
                min_decline_pct=min_downtrend_pct,
            )
            if require_downtrend and not prior_down:
                continue

            # ── Filtr: L1 musi być nowym minimum w oknie lookback ──
            # Bulkowski: "tall left side, steep decline, few or no consolidations"
            # L1 musi być najniższym Low w oknie downtrend_lookback sesji przed nim.
            # Jeśli przed L1 było niższe dno, to L1 jest tylko odbiciem od głębszego
            # minimum — nie jest prawdziwym dnem trendu spadkowego.
            # Nie stosujemy tolerancji procentowej — L1 musi być ściśle najniższym Low.
            if require_downtrend:
                lb_start = max(0, left_idx - downtrend_lookback)
                prior_lows = wdf["Low"].values[lb_start: left_idx]
                if len(prior_lows) > 0:
                    prior_min_low = float(prior_lows.min())
                    if prior_min_low < left_low:
                        continue

            # ── Filtr 4: stromy zjazd tuż przed L1 (Bulkowski "Big W / steep decline into L1") ──
            # W oknie `drop_into_l1_lookback` sesji przed L1 cena musi spaść o min min_drop_into_l1
            # od lokalnego High do Close L1. Wyklucza formacje gdzie L1 pojawia się po poziomej
            # konsolidacji lub małej korekcie — Bulkowski wymaga wyraźnego trendu spadkowego.
            if require_drop_into_l1:
                lb_start = max(0, left_idx - drop_into_l1_lookback)
                recent_window = wdf.iloc[lb_start: left_idx]
                if len(recent_window) >= 5:
                    recent_high = float(recent_window["High"].max())
                    l1_close = float(wdf.iloc[left_idx]["Close"])
                    if recent_high > 0:
                        drop = (recent_high - l1_close) / recent_high
                        if drop < min_drop_into_l1:
                            continue
                    # L1-close musi być wyraźnie poniżej średniej close w tym oknie
                    recent_avg_close = float(recent_window["Close"].mean())
                    if l1_close >= recent_avg_close:
                        continue

            # Klasyfikacja wariantu
            left_type = _classify_bottom(wdf, left_idx, window=bottom_window)
            right_type = _classify_bottom(wdf, right_idx, window=bottom_window)
            pattern_type = f"{left_type} & {right_type}"

            # Linia potwierdzenia = szczyt między dołkami (Close > peak_high)
            confirmation_price = peak_high

            # Wolumen: lewe dno powinno mieć wyższy wolumen niż prawe
            volume_ok = True
            if check_volume and "Volume" in wdf.columns:
                left_vol = float(wdf.iloc[left_idx]["Volume"])
                right_vol = float(wdf.iloc[right_idx]["Volume"])
                volume_ok = left_vol >= right_vol

            # Szukaj potwierdzenia: Close > confirmation_price po prawym dnie
            #
            # Reguły Bulkowskiego po L2:
            # 1. Close poniżej L2 price → formacja unieważniona natychmiast.
            #    Bulkowski: "48% chance price continues lower without confirming" —
            #    Close poniżej dna to wyraźny sygnał że to nie jest drugie dno.
            # 2. Okno poszukiwania breakoutu = proporcja rozpiętości L1→L2 (45%),
            #    min 5, max 60 sesji. Cena nie może błąkać się w nieskończoność.
            max_breakout_days = max(10, min(60, int(separation * 0.75)))
            search_end = min(n, right_idx + 1 + max_breakout_days)

            breakout_idx: Optional[int] = None
            for bi in range(right_idx + 1, search_end):
                close_bi = float(wdf.iloc[bi]["Close"])
                low_bi   = float(wdf.iloc[bi]["Low"])

                # Reguła 1: Close poniżej L2 → unieważnienie
                if close_bi < right_low:
                    break

                # Reguła 1b: intraday Low wyraźnie (>1%) poniżej niższego dna → unieważnienie
                if low_bi < lower_trough * 0.99:
                    break

                # Potwierdzenie: Close powyżej szczytu między dołkami
                if close_bi > confirmation_price:
                    breakout_idx = bi
                    break

            if breakout_idx is None:
                continue

            # Buduj sygnał
            left_date = wdf.index[left_idx]
            right_date = wdf.index[right_idx]
            peak_date = wdf.index[peak_pos]
            breakout_date = wdf.index[breakout_idx]
            pattern_height = confirmation_price - lower_trough

            signal = (
                f"🔻{pattern_type} "
                f"{left_date.strftime('%Y-%m-%d')}↔{right_date.strftime('%Y-%m-%d')} "
                f"(conf={confirmation_price:.2f})"
            )

            results.append(
                {
                    "date": breakout_date,
                    "signal": signal,
                    "pattern_type": pattern_type,
                    "left_trough_date": left_date,
                    "right_trough_date": right_date,
                    "peak_date": peak_date,
                    "left_trough_price": left_low,
                    "right_trough_price": right_low,
                    "peak_price": peak_high,
                    "confirmation_price": confirmation_price,
                    "pattern_height": round(pattern_height, 4),
                    "peak_rise_pct": round(peak_rise * 100, 2),
                    "bottom_diff_pct": round(bottom_diff * 100, 2),
                    "separation_days": separation,
                    "prior_downtrend": prior_down,
                    "volume_ok": volume_ok,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "breakout_idx": breakout_idx,
                }
            )

            confirmed_breakout_idx = breakout_idx
            break  # jeden prawy dołek per lewy dołek (pierwsze potwierdzenie)

    return results


# ---------------------------------------------------------------------------
# Single-bar check (używany przez backtester)
# ---------------------------------------------------------------------------

def _check_double_bottom_on_df(
    df: pd.DataFrame,
    local_min_order: int = 5,
    min_separation_days: int = 10,
    max_separation_days: int = 70,
    max_bottom_diff_pct: float = 0.06,
    min_peak_rise_pct: float = 0.10,
    require_downtrend: bool = True,
    downtrend_lookback: int = 60,
    min_downtrend_pct: float = 0.10,
    require_drop_into_l1: bool = True,
    drop_into_l1_lookback: int = 30,
    min_drop_into_l1: float = 0.05,
    bottom_window: int = 5,
    check_volume: bool = True,
) -> Optional[str]:
    """
    Zwraca sygnał jeśli OSTATNI bar df jest dniem potwierdzenia formacji.
    Używany w backtesterze (iteracja bar-po-barze).
    """
    signals = find_double_bottom_signals(
        df=df,
        local_min_order=local_min_order,
        min_separation_days=min_separation_days,
        max_separation_days=max_separation_days,
        max_bottom_diff_pct=max_bottom_diff_pct,
        min_peak_rise_pct=min_peak_rise_pct,
        require_downtrend=require_downtrend,
        downtrend_lookback=downtrend_lookback,
        min_downtrend_pct=min_downtrend_pct,
        require_drop_into_l1=require_drop_into_l1,
        drop_into_l1_lookback=drop_into_l1_lookback,
        min_drop_into_l1=min_drop_into_l1,
        bottom_window=bottom_window,
        check_volume=check_volume,
    )
    if not signals:
        return None

    last = signals[-1]
    if pd.Timestamp(last["date"]) != df.index[-1]:
        return None
    return str(last["signal"])


# ---------------------------------------------------------------------------
# Public API (analogiczne do check_flag_breakout_today)
# ---------------------------------------------------------------------------

def check_double_bottom_today(
    prices: Dict[pd.Timestamp, Dict[str, float]],
    local_min_order: int = 5,
    min_separation_days: int = 10,
    max_separation_days: int = 70,
    max_bottom_diff_pct: float = 0.06,
    min_peak_rise_pct: float = 0.10,
    require_downtrend: bool = True,
    downtrend_lookback: int = 60,
    min_downtrend_pct: float = 0.10,
    require_drop_into_l1: bool = True,
    drop_into_l1_lookback: int = 30,
    min_drop_into_l1: float = 0.05,
    bottom_window: int = 5,
    check_volume: bool = True,
) -> Optional[str]:
    """
    Sprawdź czy dzisiaj (ostatni bar) pojawia się potwierdzenie formacji podwójnego dna.
    `prices` – słownik {timestamp: {Open, High, Low, Close, Volume}}.
    """
    if not prices:
        return None
    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_double_bottom_on_df(
        df=df,
        local_min_order=local_min_order,
        min_separation_days=min_separation_days,
        max_separation_days=max_separation_days,
        max_bottom_diff_pct=max_bottom_diff_pct,
        min_peak_rise_pct=min_peak_rise_pct,
        require_downtrend=require_downtrend,
        downtrend_lookback=downtrend_lookback,
        min_downtrend_pct=min_downtrend_pct,
        require_drop_into_l1=require_drop_into_l1,
        drop_into_l1_lookback=drop_into_l1_lookback,
        min_drop_into_l1=min_drop_into_l1,
        bottom_window=bottom_window,
        check_volume=check_volume,
    )
