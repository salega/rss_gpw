from typing import Dict, Optional, List

import pandas as pd


def check_flag_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        pole_min_days: int = 3,
        pole_max_days: int = 20,
        pole_min_growth: float = 0.08,
        pole_max_daily_decline: float = 0.50,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 35,
        flag_max_retracement: float = 0.50
) -> Optional[str]:
    """
    Wykrywa formację flagi, gdzie wybicie nastąpiło w OSTATNIEJ sesji (dzisiaj):

    1. POLE (maszt): 3-12 dni wzrostu ≥8%, z dopuszczalnymi niewielkimi spadkami
       (maksymalnie połowa wzrostu z poprzedniego dnia).
       Maszt kończy się gdy MAX nie jest przebijany przez 2+ kolejne dni.
       - Z Open: używamy Close do wykrywania maksimów
       - Bez Open: używamy High do wykrywania maksimów
    2. FLAG (flaga): 3-35 dni konsolidacji od końca masztu do WCZORAJ,
       żaden zakres świecy (Open/Close lub aproksymacja) nie przekracza MAX
       ani nie spada poniżej połowy wysokości masztu
    3. BREAKOUT DZISIAJ: Close dzisiaj > MAX (wybicie w ostatniej sesji)
    """
    if not prices:
        return None

    df = pd.DataFrame.from_dict(prices, orient="index").dropna().sort_index()

    required_cols = ["Close", "High", "Low"]
    has_open = "Open" in df.columns

    if not all(col in df.columns for col in required_cols):
        return None

    if has_open:
        df = df[["Open", "Close", "High", "Low"]]
    else:
        df = df[["Close", "High", "Low"]]

    # Minimalna długość: maszt + flaga + dzień wybicia
    if len(df) < pole_min_days + flag_min_days + 1:
        return None

    # Maksymalny lookback to pole_max + flag_max + 1
    lookback_days = pole_max_days + flag_max_days_until_breakout + 1
    recent_df = df.iloc[-lookback_days:] if len(df) > lookback_days else df

    # DZISIAJ to ostatni wiersz w recent_df
    today_idx = len(recent_df) - 1
    today_close = float(recent_df.iloc[today_idx]["Close"])

    all_flags: List[Dict] = []

    # Iterujemy po możliwych punktach startowych masztu
    # Maszt musi zakończyć się co najmniej (flag_min_days + 1) dni przed dzisiaj
    max_pole_start = today_idx - flag_min_days - pole_min_days

    for pole_start_idx in range(max(0, max_pole_start - 100), max_pole_start + 1):

        for initial_pole_length in range(pole_min_days,
                                         min(pole_max_days + 1, today_idx - pole_start_idx - flag_min_days)):
            pole_end_idx = pole_start_idx + initial_pole_length - 1
            pole_window = recent_df.iloc[pole_start_idx:pole_end_idx + 1]

            closes = pole_window["Close"].values

            # Gdy brak Open, używamy High do wykrywania maksimów masztu
            if has_open:
                peaks = pole_window["Close"].values
            else:
                peaks = pole_window["High"].values

            # Warunek: pierwszy dzień nie może spadać
            if pole_start_idx > 0:
                day_before_pole = float(recent_df.iloc[pole_start_idx - 1]["Close"])
                first_close = float(closes[0])
                if first_close <= day_before_pole:
                    continue

            # Walidacja masztu
            valid_pole = True
            current_max = float(peaks[0])
            days_without_new_high = 0
            actual_pole_end_idx = pole_start_idx

            for i in range(len(closes)):
                current_close = float(closes[i])
                current_peak = float(peaks[i])

                if current_peak > current_max:
                    current_max = current_peak
                    days_without_new_high = 0
                    actual_pole_end_idx = pole_start_idx + i
                else:
                    days_without_new_high += 1

                if days_without_new_high > max_days_without_new_high:
                    break

                # Sprawdzamy czy spadek Close jest dozwolony
                if i > 0 and current_close < float(closes[i - 1]):
                    if i < 2:
                        valid_pole = False
                        break

                    day_before_prev_close = float(closes[i - 2])
                    previous_close = float(closes[i - 1])
                    previous_day_gain = previous_close - day_before_prev_close

                    if previous_day_gain <= 0:
                        valid_pole = False
                        break

                    decline = previous_close - current_close
                    max_allowed_decline = previous_day_gain * pole_max_daily_decline

                    if decline > max_allowed_decline:
                        valid_pole = False
                        break

            if not valid_pole:
                continue

            actual_pole_length = actual_pole_end_idx - pole_start_idx + 1
            if actual_pole_length < pole_min_days:
                continue

            actual_pole_window = recent_df.iloc[pole_start_idx:actual_pole_end_idx + 1]
            actual_closes = actual_pole_window["Close"].values

            pole_start_price = float(actual_closes[0])
            pole_end_price = float(actual_closes[-1])

            if pole_start_price <= 0:
                continue

            pole_growth = (pole_end_price - pole_start_price) / pole_start_price

            if pole_growth < pole_min_growth:
                continue

            max_price = current_max
            pole_height = max_price - pole_start_price

            if pole_height <= 0:
                continue

            half_pole = pole_start_price + (pole_height / 2)

            # Flaga zaczyna się po maszcie i kończy WCZORAJ (dzień przed dzisiaj)
            flag_start_idx = actual_pole_end_idx + 1
            flag_end_idx = today_idx - 1

            # Sprawdzamy czy dzisiaj Close > MAX (wybicie DZISIAJ)
            if today_close <= max_price:
                continue

            # Sprawdzamy długość flagi
            flag_length = flag_end_idx - flag_start_idx + 1

            if flag_length < flag_min_days:
                continue

            if flag_length > flag_max_days_until_breakout:
                continue

            flag_window = recent_df.iloc[flag_start_idx:flag_end_idx + 1]

            # Walidacja konsolidacji
            flag_candle_highs = []
            flag_candle_lows = []

            for idx in range(len(flag_window)):
                row = flag_window.iloc[idx]

                if has_open:
                    open_price = float(row["Open"])
                    close_price = float(row["Close"])
                    candle_high = max(open_price, close_price)
                    candle_low = min(open_price, close_price)
                else:
                    close_price = float(row["Close"])

                    if idx == 0:
                        prev_close = float(actual_pole_window.iloc[-1]["Close"])
                    else:
                        prev_close = float(flag_window.iloc[idx - 1]["Close"])

                    candle_high = max(prev_close, close_price)
                    candle_low = min(prev_close, close_price)

                flag_candle_highs.append(candle_high)
                flag_candle_lows.append(candle_low)

            # Żaden zakres świecy nie przekracza MAX
            if any(h > max_price for h in flag_candle_highs):
                continue

            # Żaden zakres świecy nie spada poniżej połowy masztu
            if any(low < half_pole for low in flag_candle_lows):
                continue

            flag_low = min(flag_candle_lows)
            retracement = (max_price - flag_low) / pole_height

            if retracement > flag_max_retracement:
                continue

            # Mamy pełną formację z wybiciem DZISIAJ!
            pole_start_date = recent_df.index[pole_start_idx]
            pole_peak_date = recent_df.index[actual_pole_end_idx]

            target_price = max_price + pole_height

            score = pole_growth * (1 - retracement) * (1 - flag_length / flag_max_days_until_breakout)

            all_flags.append({
                "pole_start_date": pole_start_date,
                "pole_peak_date": pole_peak_date,
                "max_price": max_price,
                "score": score
            })

    if not all_flags:
        return None

    # Wybieramy najlepszą formację
    best = max(all_flags, key=lambda x: x["score"])

    pole_start_str = best["pole_start_date"].strftime("%Y-%m-%d")
    pole_peak_str = best["pole_peak_date"].strftime("%Y-%m-%d")

    return f"🚩{pole_start_str} → {pole_peak_str} ({best['max_price']:.2f})"
