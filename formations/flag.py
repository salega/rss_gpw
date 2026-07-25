import pandas as pd
from typing import Dict, Optional, List


def _prepare_flag_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Close", "High", "Low"]
    has_open = "Open" in df.columns
    has_volume = "Volume" in df.columns

    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    base_cols = ["Open", "Close", "High", "Low"] if has_open else ["Close", "High", "Low"]
    if has_volume:
        base_cols = base_cols + ["Volume"]

    return df[base_cols].dropna(subset=required_cols).sort_index()


def _build_signal(pole_start_date: pd.Timestamp, pole_peak_date: pd.Timestamp, max_price: float) -> str:
    pole_start_str = pole_start_date.strftime("%Y-%m-%d")
    pole_peak_str = pole_peak_date.strftime("%Y-%m-%d")
    return f"🚩{pole_start_str} → {pole_peak_str} ({max_price:.2f})"


def _get_candle_body_high_low(
        has_open: bool,
        open_price: Optional[float],
        close_price: float,
        prev_close: float,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
) -> tuple[float, float]:
    # Bulkowski używa rzeczywistych High/Low świecy, nie tylko ciała
    if high_price is not None and low_price is not None:
        return high_price, low_price

    if has_open and open_price is not None:
        return max(open_price, close_price), min(open_price, close_price)

    return max(prev_close, close_price), min(prev_close, close_price)


def _check_volume_declining_in_flag(working_df: pd.DataFrame, flag_start_idx: int, flag_end_idx: int) -> bool:
    """Sprawdza czy wolumen maleje w czasie formowania flagi (kryterium Bulkowskiego)."""
    if "Volume" not in working_df.columns:
        return True  # brak danych wolumenu – nie odrzucamy

    flag_volumes = working_df.iloc[flag_start_idx:flag_end_idx + 1]["Volume"].dropna().tolist()
    if len(flag_volumes) < 3:
        return True

    # Regresja liniowa nachylenia wolumenu – ujemne = maleje
    import numpy as np
    x = list(range(len(flag_volumes)))
    slope = float(np.polyfit(x, flag_volumes, 1)[0])
    return slope <= 0


def _check_flag_density(working_df: pd.DataFrame, flag_start_idx: int, flag_end_idx: int) -> bool:
    """Sprawdza gęstość flagi: czy kolejne słupki zachodzą na siebie (kryterium Bulkowskiego).
    Gęsta flaga = większość sąsiadujących par słupków ma nakładające się zakresy High/Low."""
    if flag_end_idx <= flag_start_idx:
        return False

    overlaps = 0
    total = 0
    for i in range(flag_start_idx, flag_end_idx):
        h1 = float(working_df.iloc[i]["High"])
        l1 = float(working_df.iloc[i]["Low"])
        h2 = float(working_df.iloc[i + 1]["High"])
        l2 = float(working_df.iloc[i + 1]["Low"])
        if l2 <= h1 and l1 <= h2:  # zakresy się nakładają
            overlaps += 1
        total += 1

    if total == 0:
        return True
    return (overlaps / total) >= 0.5  # co najmniej połowa par zachodzi na siebie


def _find_pole_from_index(
        working_df: pd.DataFrame,
        start_idx: int,
        pole_min_days: int,
        pole_max_days: int,
        pole_min_growth: float,
        pole_max_daily_decline: float,
        max_days_without_new_high: int,  # zachowane dla kompatybilności, nieużywane
) -> Optional[dict]:
    n = len(working_df)

    if start_idx >= n - pole_min_days:
        return None

    if start_idx > 0:
        first_close = float(working_df.iloc[start_idx]["Close"])
        day_before_close = float(working_df.iloc[start_idx - 1]["Close"])
        if first_close <= day_before_close:
            return None

    pole_start_close = float(working_df.iloc[start_idx]["Close"])
    if pole_start_close <= 0:
        return None

    max_end_idx = min(n - 1, start_idx + pole_max_days - 1)

    # Bulkowski: maszt = okno w którym kurs prawie się podwaja.
    # Szukamy najwyższego High w oknie i sprawdzamy czy wzrost ≥ pole_min_growth.
    # Jedyny warunek odrzucający w trakcie: głęboka korekta (sygnał że trend się skończył).
    current_max = float(working_df.iloc[start_idx]["High"])
    actual_pole_end_idx = start_idx
    closes: List[float] = []

    days_without_new_high = 0

    for idx in range(start_idx, max_end_idx + 1):
        row = working_df.iloc[idx]
        current_close = float(row["Close"])
        current_high = float(row["High"])
        closes.append(current_close)

        if current_high > current_max:
            current_max = current_high
            actual_pole_end_idx = idx
            days_without_new_high = 0
        else:
            days_without_new_high += 1

        # Bulkowski: maszt to impulsowy wzrost — kończy się gdy momentum ustaje.
        # Zatrzymaj maszt po max_days_without_new_high dniach bez nowego High.
        if days_without_new_high >= max_days_without_new_high:
            break

        # Odrzuć maszt jeśli Close spada poniżej startu (wzrost wyzerowany)
        if current_close < pole_start_close:
            return None

    actual_pole_length = actual_pole_end_idx - start_idx + 1
    if actual_pole_length < pole_min_days:
        return None

    pole_start_price = float(working_df.iloc[start_idx]["Close"])
    pole_end_price = float(working_df.iloc[actual_pole_end_idx]["Close"])

    if pole_start_price <= 0:
        return None

    # Bulkowski: wzrost masztu liczony od Close startu do High szczytu
    max_price = current_max  # = High dnia szczytu
    pole_growth = (max_price - pole_start_price) / pole_start_price
    if pole_growth < pole_min_growth:
        return None

    pole_height = max_price - pole_start_price
    if pole_height <= 0:
        return None

    return {
        "pole_start_idx": start_idx,
        "pole_end_idx": actual_pole_end_idx,
        "pole_start_date": working_df.index[start_idx],
        "pole_peak_date": working_df.index[actual_pole_end_idx],
        "pole_start_price": float(pole_start_price),
        "pole_end_price": float(pole_end_price),
        "pole_growth": float(pole_growth),
        "max_price": float(max_price),
        "pole_height": float(pole_height),
    }


def _find_breakout_after_pole(
        working_df: pd.DataFrame,
        pole: dict,
        flag_min_days: int,
        flag_max_days_until_breakout: int,
        require_volume_decline: bool = True,
        require_dense_flag: bool = False,
) -> Optional[dict]:
    has_open = "Open" in working_df.columns
    n = len(working_df)

    pole_end_idx = int(pole["pole_end_idx"])
    pole_start_price = float(pole["pole_start_price"])
    max_price = float(pole["max_price"])
    pole_height = float(pole["pole_height"])

    half_pole = pole_start_price + (pole_height / 2.0)

    flag_start_idx = pole_end_idx + 1
    if flag_start_idx >= n:
        return None

    flag_candle_highs: List[float] = []
    flag_candle_lows: List[float] = []

    for idx in range(flag_start_idx, min(n, pole_end_idx + 1 + flag_max_days_until_breakout + 1)):
        close_price = float(working_df.iloc[idx]["Close"])

        if idx == flag_start_idx:
            prev_close = float(working_df.iloc[pole_end_idx]["Close"])
        else:
            prev_close = float(working_df.iloc[idx - 1]["Close"])

        open_price = float(working_df.iloc[idx]["Open"]) if has_open else None
        high_price = float(working_df.iloc[idx]["High"])
        low_price = float(working_df.iloc[idx]["Low"])

        candle_high, candle_low = _get_candle_body_high_low(
            has_open=has_open,
            open_price=open_price,
            close_price=close_price,
            prev_close=prev_close,
            high_price=high_price,
            low_price=low_price,
        )

        days_in_flag = idx - flag_start_idx + 1

        flag_high_so_far = max(flag_candle_highs) if flag_candle_highs else 0.0

        # Bulkowski: wybicie = Close powyżej najwyższego szczytu flagi (nie szczytu masztu)
        if close_price > flag_high_so_far and flag_high_so_far > 0 and days_in_flag > 1:
            if days_in_flag < flag_min_days:
                return None

            flag_end_idx = idx - 1
            breakout_idx = idx

            if flag_end_idx < flag_start_idx:
                return None

            if len(flag_candle_lows) < flag_min_days:
                return None

            # Flaga nie może wyjść powyżej szczytu masztu przed wybiciu
            if any(h > max_price for h in flag_candle_highs):
                return None

            if any(low < half_pole for low in flag_candle_lows):
                return None

            flag_low = min(flag_candle_lows)
            retracement = (max_price - flag_low) / pole_height

            # Bulkowski: cofnięcie 0–25% wysokości masztu
            # (IIIN miał ~0-7%, więc nie wymuszamy minimum 10%)
            if retracement > 0.25:
                return None

            # Bulkowski: wolumen powinien maleć podczas flagi
            if require_volume_decline:
                if not _check_volume_declining_in_flag(working_df, flag_start_idx, flag_end_idx):
                    return None

            # Bulkowski: gęsta flaga (słupki zachodzą na siebie) – opcjonalnie
            if require_dense_flag:
                if not _check_flag_density(working_df, flag_start_idx, flag_end_idx):
                    return None

            flag_high = max(flag_candle_highs)
            score = float(pole["pole_growth"]) * (1 - retracement) * (
                    1 - len(flag_candle_lows) / flag_max_days_until_breakout
            )

            return {
                **pole,
                "flag_start_idx": flag_start_idx,
                "flag_end_idx": flag_end_idx,
                "flag_low": float(flag_low),
                "flag_high": float(flag_high),
                "retracement": float(retracement),
                "breakout_idx": breakout_idx,
                "breakout_date": working_df.index[breakout_idx],
                "score": float(score),
            }

        flag_candle_highs.append(candle_high)
        flag_candle_lows.append(candle_low)

        # Odrzuć jeśli flaga spada za nisko (poniżej połowy masztu)
        if candle_low < half_pole:
            return None

        # Odrzuć jeśli cofnięcie już przekroczyło 25% masztu
        flag_low = min(flag_candle_lows)
        retracement = (max_price - flag_low) / pole_height
        if retracement > 0.25:
            return None

    return None


def find_flag_breakouts_on_df(
        df: pd.DataFrame,
        # --- Maszt (Bulkowski: wzrost ≥90% w ≤40 sesjach) ---
        pole_min_days: int = 4,
        pole_max_days: int = 40,           # 2 miesiące sesyjne
        pole_min_growth: float = 0.90,     # ≥90% wzrostu (WWF Bulkowskiego)
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        # --- Flaga (Bulkowski: 5–19 dni, cofnięcie 10–25%) ---
        flag_min_days: int = 5,
        flag_max_days_until_breakout: int = 19,  # Bulkowski: max 19 dni
        require_volume_decline: bool = True,      # Bulkowski: wolumen maleje w fladze
        require_dense_flag: bool = False,         # Bulkowski: gęsta flaga (wyższe wybicia)
) -> List[dict]:
    working_df = _prepare_flag_df(df)
    if working_df.empty:
        return []

    min_required_len = pole_min_days + flag_min_days + 1
    if len(working_df) < min_required_len:
        return []

    results: List[dict] = []
    i = 0
    n = len(working_df)

    while i < n - min_required_len + 1:
        pole = _find_pole_from_index(
            working_df=working_df,
            start_idx=i,
            pole_min_days=pole_min_days,
            pole_max_days=pole_max_days,
            pole_min_growth=pole_min_growth,
            pole_max_daily_decline=pole_max_daily_decline,
            max_days_without_new_high=max_days_without_new_high,
        )

        if pole is None:
            i += 1
            continue

        breakout = _find_breakout_after_pole(
            working_df=working_df,
            pole=pole,
            flag_min_days=flag_min_days,
            flag_max_days_until_breakout=flag_max_days_until_breakout,
            require_volume_decline=require_volume_decline,
            require_dense_flag=require_dense_flag,
        )

        if breakout is None:
            i += 1
            continue

        signal = _build_signal(
            pole_start_date=breakout["pole_start_date"],
            pole_peak_date=breakout["pole_peak_date"],
            max_price=float(breakout["max_price"]),
        )

        results.append(
            {
                "date": breakout["breakout_date"],
                "signal": signal,
                "pole_start_date": breakout["pole_start_date"],
                "pole_peak_date": breakout["pole_peak_date"],
                "pole_growth_pct": round(float(breakout["pole_growth"]) * 100, 2),
                "max_price": float(breakout["max_price"]),
                "flag_low": float(breakout["flag_low"]),
                "flag_high": float(breakout["flag_high"]),
                "retracement_pct": round(float(breakout["retracement"]) * 100, 2),
                "flag_days": int(breakout["breakout_idx"]) - int(breakout["flag_start_idx"]),
                "score": float(breakout["score"]),
            }
        )

        i = int(breakout["breakout_idx"]) + 1

    return results


def _check_flag_breakout_on_df(
        df: pd.DataFrame,
        pole_min_days: int = 4,
        pole_max_days: int = 40,
        pole_min_growth: float = 0.90,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 5,
        flag_max_days_until_breakout: int = 19,
        require_volume_decline: bool = True,
        require_dense_flag: bool = False,
        breakout_idx: Optional[int] = None,
) -> Optional[str]:
    working_df = _prepare_flag_df(df)
    if working_df.empty:
        return None

    if breakout_idx is not None:
        if breakout_idx < 0 or breakout_idx >= len(working_df):
            return None
        working_df = working_df.iloc[:breakout_idx + 1]

    found = find_flag_breakouts_on_df(
        df=working_df,
        pole_min_days=pole_min_days,
        pole_max_days=pole_max_days,
        pole_min_growth=pole_min_growth,
        pole_max_daily_decline=pole_max_daily_decline,
        max_days_without_new_high=max_days_without_new_high,
        flag_min_days=flag_min_days,
        flag_max_days_until_breakout=flag_max_days_until_breakout,
        require_volume_decline=require_volume_decline,
        require_dense_flag=require_dense_flag,
    )

    if not found:
        return None

    last_signal = found[-1]
    if pd.Timestamp(last_signal["date"]) != working_df.index[-1]:
        return None

    return str(last_signal["signal"])


def check_flag_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        pole_min_days: int = 4,
        pole_max_days: int = 40,
        pole_min_growth: float = 0.90,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 5,
        flag_max_days_until_breakout: int = 19,
        require_volume_decline: bool = True,
        require_dense_flag: bool = False,
) -> Optional[str]:
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index")
    return _check_flag_breakout_on_df(
        df=df,
        pole_min_days=pole_min_days,
        pole_max_days=pole_max_days,
        pole_min_growth=pole_min_growth,
        pole_max_daily_decline=pole_max_daily_decline,
        max_days_without_new_high=max_days_without_new_high,
        flag_min_days=flag_min_days,
        flag_max_days_until_breakout=flag_max_days_until_breakout,
        require_volume_decline=require_volume_decline,
        require_dense_flag=require_dense_flag,
    )