"""
Detekcja formacji Bump-and-Run Reversal Bottom (BRRB)
wg Thomasa Bulkowskiego (odkryta 1999).

Kształt: patelnia (frying pan) przechylona w dół, rączka po lewej.
Trzy fazy:
  1. Lead-in phase  — łagodny trend spadkowy (0–45°), rączka patelni
  2. Bump phase     — gwałtowny, stromy spadek (≥60°), dno zaokrąglone
  3. Run phase      — odbicie i wybicie powyżej linii trendu (potwierdzenie)

Kluczowe miary:
  - lead_in_height: najdalsze odchylenie ceny od linii trendu w lead-in (1. ćwiartce)
  - bump_height:    najdalsze odchylenie w bump phase — musi być ≥ 2× lead_in_height
  - Potwierdzenie:  Close > linia trendu (lead-in trendline)

Wyniki (bull market, 1,099 transakcji):
  - Rank: 1/39 (najlepsza formacja!)
  - Break even failure: 9%
  - Avg rise: 55%
  - Throwback rate: 61%
  - Price target met: 76%
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


def _fit_trendline(highs: np.ndarray) -> Tuple[float, float]:
    """
    Dopasuj linię trendu do High (regresja liniowa).
    Zwraca (slope, intercept) — linia trendu = slope*x + intercept.
    """
    x = np.arange(len(highs), dtype=float)
    slope, intercept = np.polyfit(x, highs, 1)
    return float(slope), float(intercept)


def _trendline_value(slope: float, intercept: float, x: int) -> float:
    return slope * x + intercept


def _trendline_angle_deg(slope: float, price_scale: float = 1.0) -> float:
    """
    Kąt linii trendu w stopniach.
    Bulkowski używa skali arytmetycznej i mierzy kąt wizualnie.
    Normalizujemy slope do % ceny/sesję żeby kąt był porównywalny.
    slope_pct = slope / price_scale = zmiana procentowa ceny na sesję.
    Przy slope_pct = -0.01 (-1%/sesję) kąt ≈ 45° (steep decline).
    """
    if price_scale <= 0:
        return 0.0
    slope_pct = slope / price_scale  # zmiana jako % ceny na sesję
    # Skalujemy: zakładamy że -1%/sesję = 45°, -2%/sesję ≈ 63°
    # tan(45°) = 1, więc normalizujemy slope_pct × 100 (w %)
    return float(abs(np.degrees(np.arctan(abs(slope_pct) * 100))))


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def find_bump_and_run_signals(
    df: pd.DataFrame,
    # --- Lead-in phase ---
    min_lead_in_days: int = 35,         # Bulkowski avg 35 dni
    max_lead_in_days: int = 120,
    max_lead_in_angle: float = 45.0,    # max kąt linii trendu w lead-in (0–45°)
    # --- Bump phase ---
    min_bump_days: int = 10,            # min długość bumpa
    max_bump_days: int = 90,
    min_bump_angle: float = 60.0,       # min kąt trendu w bump (≥60°)
    min_bump_height_ratio: float = 2.0, # Bulkowski: bump ≥ 2× lead-in height
    # --- Run phase / breakout ---
    max_breakout_days: int = 90,        # ABT: ~2 miesiące od dna do breakoutu
    # --- Ogólne ---
    local_order: int = 25,              # okno dla lokalnych ekstremów — tylko dominujące minima
) -> List[dict]:
    """
    Skanuj DataFrame i zwróć listę potwierdzonych formacji BRRB.

    Każdy rekord zawiera:
      date              – data potwierdzenia (Close > trendline)
      signal            – opis tekstowy
      lead_in_start     – początek lead-in phase
      lead_in_end       – koniec lead-in / początek bump
      bump_low_date     – dno bumpa (najniższy Low)
      breakout_date     – data potwierdzenia
      trendline_slope   – nachylenie linii trendu lead-in
      trendline_at_breakout – wartość linii trendu w dniu wybicia
      lead_in_height    – wysokość lead-in (odchylenie od trendline)
      bump_height       – wysokość bumpa (odchylenie od trendline)
      bump_height_ratio – bump_height / lead_in_height
      lead_in_days      – długość lead-in
      bump_days         – długość bumpa
      breakout_days     – dni od dna bumpa do wybicia
      pattern_high      – najwyższy High w całej formacji (cel cenowy wg TB)
    """
    wdf = _prepare_df(df)
    if wdf.empty or len(wdf) < min_lead_in_days + min_bump_days + 5:
        return []

    highs  = wdf["High"].values
    lows   = wdf["Low"].values
    closes = wdf["Close"].values
    n = len(wdf)
    results: List[dict] = []
    confirmed_breakout_idx = -1

    # --- Strategia O(k²) zamiast O(n³) ---
    # 1. Znajdź kandydatów na dno bumpa (lokalne minima Low z dużym oknem)
    # 2. Dla każdego dna bumpa szukaj lead-in wstecz
    # 3. Szukaj breakoutu w przód

    # Lokalne minima jako kandydaci na bump_low
    bump_candidates: List[int] = []
    for i in range(local_order, n - local_order):
        window_lows = lows[i - local_order: i + local_order + 1]
        if lows[i] == window_lows.min():
            bump_candidates.append(i)

    for bump_low_idx in bump_candidates:
        # Po znalezieniu sygnału, następny bump musi zacząć się znacznie później
        # (co najmniej min_lead_in_days po poprzednim breakoucie)
        if bump_low_idx <= confirmed_breakout_idx + min_lead_in_days:
            continue

        bump_low_val = float(lows[bump_low_idx])

        # Dla każdego bump_low szukamy JEDNEJ najlepszej kombinacji (li_end, li_start).
        # Kryteria: największy bump_ratio przy spełnieniu wszystkich warunków.
        best_candidate: Optional[dict] = None

        for li_end in range(
            max(local_order, bump_low_idx - max_bump_days),
            max(local_order, bump_low_idx - min_bump_days)
        ):
            if float(lows[li_end]) <= bump_low_val * 1.05:
                continue

            bump_days = bump_low_idx - li_end
            if not (min_bump_days <= bump_days <= max_bump_days):
                continue

            bump_highs_seg = highs[li_end: bump_low_idx + 1]
            if len(bump_highs_seg) < 3:
                continue
            bump_slope, _ = _fit_trendline(bump_highs_seg)
            if bump_slope >= 0:
                continue
            price_scale = float(np.mean(bump_highs_seg))
            bump_angle = _trendline_angle_deg(bump_slope, price_scale)
            if bump_angle < min_bump_angle:
                continue

            # Jeden li_start per li_end — wybierz najdłuższy pasujący lead-in
            best_li_start: Optional[int] = None
            best_li_data: Optional[dict] = None

            for li_start in range(
                max(0, li_end - max_lead_in_days),
                max(0, li_end - min_lead_in_days)
            ):
                lead_in_days = li_end - li_start
                if not (min_lead_in_days <= lead_in_days <= max_lead_in_days):
                    continue

                li_highs = highs[li_start: li_end + 1]
                if len(li_highs) < 5:
                    continue

                slope, intercept = _fit_trendline(li_highs)
                if slope >= 0:
                    continue
                li_mean = float(np.mean(li_highs))
                li_angle = _trendline_angle_deg(slope, li_mean)
                if li_angle > max_lead_in_angle:
                    continue
                if li_angle > 0 and bump_angle < li_angle * 1.5:
                    continue

                quarter = max(3, lead_in_days // 4)
                li_height = 0.0
                for j in range(min(quarter, li_end - li_start)):
                    tl_val = _trendline_value(slope, intercept, j)
                    dist = tl_val - float(lows[li_start + j])
                    if dist > li_height:
                        li_height = dist

                if li_height <= 0:
                    continue

                bump_height = 0.0
                for j in range(bump_low_idx - li_end + 1):
                    x_abs = (li_end - li_start) + j
                    tl_val = _trendline_value(slope, intercept, x_abs)
                    dist = tl_val - float(lows[li_end + j])
                    if dist > bump_height:
                        bump_height = dist

                if bump_height < min_bump_height_ratio * li_height:
                    continue

                bump_ratio_cand = bump_height / li_height if li_height > 0 else 0.0
                # Zachowaj najlepszy (największy ratio, najdłuższy lead-in)
                if best_li_data is None or bump_ratio_cand > best_li_data["bump_ratio"]:
                    best_li_start = li_start
                    best_li_data = {
                        "li_start": li_start, "li_end": li_end,
                        "slope": slope, "intercept": intercept,
                        "li_height": li_height, "bump_height": bump_height,
                        "bump_ratio": bump_ratio_cand, "bump_days": bump_days,
                        "lead_in_days": lead_in_days,
                    }
                break  # weź pierwszy pasujący li_start (najdłuższy lead-in) i idź dalej

            if best_li_data is None:
                continue

            # Sprawdź breakout dla najlepszej kombinacji
            slope     = best_li_data["slope"]
            intercept = best_li_data["intercept"]
            li_start  = best_li_data["li_start"]

            search_start = bump_low_idx + 1
            search_end   = min(n, bump_low_idx + max_breakout_days + 1)

            breakout_idx: Optional[int] = None
            for bi in range(search_start, search_end):
                x_abs    = bi - li_start
                tl_val   = _trendline_value(slope, intercept, x_abs)
                close_bi = float(closes[bi])
                if float(lows[bi]) < bump_low_val * 0.95:
                    break
                if close_bi > tl_val:
                    breakout_idx = bi
                    break

            if breakout_idx is None:
                continue

            # Zachowaj jako kandydata dla tego bump_low
            if best_candidate is None or best_li_data["bump_ratio"] > best_candidate["bump_ratio"]:
                best_candidate = {**best_li_data, "breakout_idx": breakout_idx}
            break  # jeden li_end per bump_low

        if best_candidate is None:
            continue

        # Używamy best_candidate do budowy sygnału
        li_start     = best_candidate["li_start"]
        li_end       = best_candidate["li_end"]
        slope        = best_candidate["slope"]
        intercept    = best_candidate["intercept"]
        breakout_idx = best_candidate["breakout_idx"]
        bump_height  = best_candidate["bump_height"]
        li_height    = best_candidate["li_height"]
        li_start_date  = wdf.index[li_start]
        li_end_date    = wdf.index[li_end]
        bump_low_date  = wdf.index[bump_low_idx]
        breakout_date  = wdf.index[breakout_idx]
        tl_at_breakout = _trendline_value(slope, intercept, breakout_idx - li_start)
        pattern_high   = float(highs[li_start: breakout_idx + 1].max())
        bump_ratio     = bump_height / li_height if li_height > 0 else 0.0
        breakout_days  = breakout_idx - bump_low_idx

        signal = (
            f"🍳{li_start_date.strftime('%Y-%m-%d')}→{bump_low_date.strftime('%Y-%m-%d')}"
            f"→{breakout_date.strftime('%Y-%m-%d')} "
            f"(ratio={bump_ratio:.1f}×, tl={tl_at_breakout:.2f})"
        )

        results.append({
            "date":                  breakout_date,
            "signal":                signal,
            "lead_in_start_date":    li_start_date,
            "lead_in_end_date":      li_end_date,
            "bump_low_date":         bump_low_date,
            "breakout_date":         breakout_date,
            "trendline_slope":       round(slope, 6),
            "trendline_at_breakout": round(tl_at_breakout, 4),
            "lead_in_height":        round(li_height, 4),
            "bump_height":           round(bump_height, 4),
            "bump_height_ratio":     round(bump_ratio, 2),
            "lead_in_days":          best_candidate["lead_in_days"],
            "bump_days":             best_candidate["bump_days"],
            "breakout_days":         breakout_days,
            "pattern_high":          round(pattern_high, 4),
            "li_start_idx":          li_start,
            "bump_low_idx":          bump_low_idx,
            "breakout_idx":          breakout_idx,
        })

        confirmed_breakout_idx = breakout_idx

    # Sortuj po dacie i usuń duplikaty
    results.sort(key=lambda r: r["breakout_date"])
    seen: set = set()
    unique: List[dict] = []
    for r in results:
        key = r["breakout_idx"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ---------------------------------------------------------------------------
# Single-bar check
# ---------------------------------------------------------------------------

def _check_bump_and_run_on_df(
    df: pd.DataFrame,
    min_lead_in_days: int = 35,
    max_lead_in_days: int = 120,
    max_lead_in_angle: float = 45.0,
    min_bump_days: int = 10,
    max_bump_days: int = 90,
    min_bump_angle: float = 60.0,
    min_bump_height_ratio: float = 2.0,
    max_breakout_days: int = 90,
    local_order: int = 25,
) -> Optional[str]:
    signals = find_bump_and_run_signals(
        df=df,
        min_lead_in_days=min_lead_in_days,
        max_lead_in_days=max_lead_in_days,
        max_lead_in_angle=max_lead_in_angle,
        min_bump_days=min_bump_days,
        max_bump_days=max_bump_days,
        min_bump_angle=min_bump_angle,
        min_bump_height_ratio=min_bump_height_ratio,
        max_breakout_days=max_breakout_days,
        local_order=local_order,
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

def check_bump_and_run_today(
    prices: Dict[pd.Timestamp, Dict[str, float]],
    min_lead_in_days: int = 35,
    max_lead_in_days: int = 120,
    max_lead_in_angle: float = 45.0,
    min_bump_days: int = 10,
    max_bump_days: int = 90,
    min_bump_angle: float = 60.0,
    min_bump_height_ratio: float = 2.0,
    max_breakout_days: int = 90,
    local_order: int = 25,
) -> Optional[str]:
    """
    Sprawdź czy dzisiaj (ostatni bar) pojawia się potwierdzenie BRRB.
    `prices` – słownik {timestamp: {Open, High, Low, Close, Volume}}.
    """
    if not prices:
        return None
    df = pd.DataFrame.from_dict(prices, orient="index").sort_index()
    return _check_bump_and_run_on_df(
        df=df,
        min_lead_in_days=min_lead_in_days,
        max_lead_in_days=max_lead_in_days,
        max_lead_in_angle=max_lead_in_angle,
        min_bump_days=min_bump_days,
        max_bump_days=max_bump_days,
        min_bump_angle=min_bump_angle,
        min_bump_height_ratio=min_bump_height_ratio,
        max_breakout_days=max_breakout_days,
        local_order=local_order,
    )
