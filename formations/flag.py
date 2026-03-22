from typing import Dict, Optional, List

import pandas as pd


def check_flag_breakout_today(
        prices: Dict[pd.Timestamp, Dict[str, float]],
        lookback_days: int = 200,
        pole_min_days: int = 3,
        pole_max_days: int = 12,
        pole_min_growth: float = 0.08,
        pole_max_daily_decline: float = 0.50,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 35,
        flag_max_retracement: float = 0.50
) -> Optional[str]:
    """
    Wykrywa formację flagi w ostatnich lookback_days:

    1. POLE (maszt): 3-12 dni wzrostu ≥8%, z dopuszczalnymi niewielkimi spadkami
       (maksymalnie połowa wzrostu z poprzedniego dnia).
       Maszt kończy się gdy MAX nie jest przebijany przez 2+ kolejne dni.
       - Z Open: używamy Close do wykrywania maksimów
       - Bez Open: używamy High do wykrywania maksimów
    2. FLAG (flaga): 3-35 dni od startu konsolidacji do wybicia,
       żaden zakres świecy (Open/Close lub aproksymacja) nie przekracza MAX
       ani nie spada poniżej połowy wysokości masztu
    3. BREAKOUT: wybicie powyżej MAX w ciągu 35 dni od startu flagi (Close > MAX)
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

    if len(df) < lookback_days:
        return None

    recent_df = df.iloc[-lookback_days:]
    all_flags: List[Dict] = []

    for pole_start_idx in range(len(recent_df) - pole_min_days - flag_min_days):

        for initial_pole_length in range(pole_min_days,
                                         min(pole_max_days + 1, len(recent_df) - pole_start_idx - flag_min_days)):
            pole_end_idx = pole_start_idx + initial_pole_length - 1
            pole_window = recent_df.iloc[pole_start_idx:pole_end_idx + 1]

            closes = pole_window["Close"].values

            # Gdy brak Open, używamy High do wykrywania maksimów masztu
            if has_open:
                peaks = pole_window["Close"].values  # Close = rzeczywisty szczyt ciała świecy
            else:
                peaks = pole_window["High"].values  # High = szczyt z knotem

            # Warunek: pierwszy dzień nie może spadać (względem dnia przed pole_start_idx, jeśli istnieje)
            if pole_start_idx > 0:
                day_before_pole = float(recent_df.iloc[pole_start_idx - 1]["Close"])
                first_close = float(closes[0])
                if first_close <= day_before_pole:
                    continue

            # Warunek: sprawdzamy czy maszt rośnie z dopuszczalnymi niewielkimi spadkami
            valid_pole = True
            current_max = float(peaks[0])
            days_without_new_high = 0
            actual_pole_end_idx = pole_start_idx

            for i in range(len(closes)):
                current_close = float(closes[i])
                current_peak = float(peaks[i])

                # Sprawdzamy czy ustanowiono nowy MAX
                if current_peak > current_max:
                    current_max = current_peak
                    days_without_new_high = 0
                    actual_pole_end_idx = pole_start_idx + i
                else:
                    days_without_new_high += 1

                # Jeśli mamy więcej niż max_days_without_new_high dni bez nowego MAX, kończymy maszt
                if days_without_new_high > max_days_without_new_high:
                    break

                # Sprawdzamy czy spadek Close jest dozwolony (jeśli nie jest to nowy MAX)
                if i > 0 and current_close < float(closes[i - 1]):
                    if i < 2:
                        # Drugi dzień nie może spadać
                        valid_pole = False
                        break

                    day_before_prev_close = float(closes[i - 2])
                    previous_close = float(closes[i - 1])
                    previous_day_gain = previous_close - day_before_prev_close

                    if previous_day_gain <= 0:
                        # Jeśli poprzedni dzień spadał, obecny dzień też nie może spadać
                        valid_pole = False
                        break

                    # Dozwolony spadek to maksymalnie połowa wzrostu z poprzedniego dnia
                    decline = previous_close - current_close
                    max_allowed_decline = previous_day_gain * pole_max_daily_decline

                    if decline > max_allowed_decline:
                        valid_pole = False
                        break

            if not valid_pole:
                continue

            # Teraz actual_pole_end_idx wskazuje na rzeczywisty koniec masztu
            actual_pole_length = actual_pole_end_idx - pole_start_idx + 1
            if actual_pole_length < pole_min_days:
                continue

            actual_pole_window = recent_df.iloc[pole_start_idx:actual_pole_end_idx + 1]
            actual_closes = actual_pole_window["Close"].values

            # Warunek: całkowity wzrost >= pole_min_growth (8%)
            pole_start_price = float(actual_closes[0])
            pole_end_price = float(actual_closes[-1])

            if pole_start_price <= 0:
                continue

            pole_growth = (pole_end_price - pole_start_price) / pole_start_price

            if pole_growth < pole_min_growth:
                continue

            # MAX to najwyższy szczyt w maszcie (Close lub High, zależnie od has_open)
            max_price = current_max
            pole_height = max_price - pole_start_price

            if pole_height <= 0:
                continue

            # Próg: połowa wysokości masztu
            half_pole = pole_start_price + (pole_height / 2)

            # Teraz szukamy flagi po maszcie
            flag_start_idx = actual_pole_end_idx + 1

            if flag_start_idx >= len(recent_df):
                continue

            # Szukamy wybicia w ciągu flag_max_days_until_breakout dni od startu flagi
            max_search_idx = min(flag_start_idx + flag_max_days_until_breakout, len(recent_df))

            # Sprawdzamy, czy w tym oknie jest wybicie (Close > MAX)
            breakout_idx = None
            breakout_price = None

            for idx in range(flag_start_idx, max_search_idx):
                close_price = float(recent_df.iloc[idx]["Close"])
                if close_price > max_price:
                    breakout_idx = idx
                    breakout_price = close_price
                    break

            # Jeśli nie było wybicia w ciągu flag_max_days_until_breakout dni, pomijamy
            if breakout_idx is None:
                continue

            # Teraz sprawdzamy okres konsolidacji (od flag_start do breakout-1)
            flag_end_idx = breakout_idx - 1

            # Musi być minimum flag_min_days konsolidacji
            flag_length = flag_end_idx - flag_start_idx + 1
            if flag_length < flag_min_days:
                continue

            flag_window = recent_df.iloc[flag_start_idx:flag_end_idx + 1]

            # Dla każdego dnia w fladze sprawdzamy zakres świecy
            flag_candle_highs = []
            flag_candle_lows = []

            for idx in range(len(flag_window)):
                row = flag_window.iloc[idx]

                if has_open:
                    # Używamy Open i Close
                    open_price = float(row["Open"])
                    close_price = float(row["Close"])

                    # Górna krawędź świecy to max(Open, Close)
                    candle_high = max(open_price, close_price)
                    # Dolna krawędź świecy to min(Open, Close)
                    candle_low = min(open_price, close_price)
                else:
                    # Fallback: używamy tylko Close (knoty ignorujemy)
                    # Zakładamy, że poprzednie Close ≈ dzisiejsze Open
                    close_price = float(row["Close"])

                    if idx == 0:
                        # Pierwszy dzień flagi: porównujemy z końcem masztu
                        prev_close = float(actual_pole_window.iloc[-1]["Close"])
                    else:
                        # Kolejne dni: poprzednie Close
                        prev_close = float(flag_window.iloc[idx - 1]["Close"])

                    candle_high = max(prev_close, close_price)
                    candle_low = min(prev_close, close_price)

                flag_candle_highs.append(candle_high)
                flag_candle_lows.append(candle_low)

            # Warunek: żaden zakres świecy nie przekracza MAX
            if any(h > max_price for h in flag_candle_highs):
                continue

            # Warunek: żaden zakres świecy nie spada poniżej połowy wysokości masztu
            if any(low < half_pole for low in flag_candle_lows):
                continue

            # Oblicz retracement (jak głęboko spadła flaga od MAX)
            flag_low = min(flag_candle_lows)
            retracement = (max_price - flag_low) / pole_height

            # Warunek: spadek maksymalnie o 50% wysokości masztu
            if retracement > flag_max_retracement:
                continue

            # Mamy pełną formację: pole + flag + breakout!
            pole_start_date = recent_df.index[pole_start_idx]
            flag_start_date = recent_df.index[flag_start_idx]
            breakout_date = recent_df.index[breakout_idx]
            days_since_breakout = len(recent_df) - 1 - breakout_idx

            # Target: MAX + wysokość masztu
            target_price = max_price + pole_height

            # Score: im większy wzrost masztu, im mniejszy retracement, im szybsze wybicie, tym lepiej
            score = pole_growth * (1 - retracement) * (1 - flag_length / flag_max_days_until_breakout)

            all_flags.append({
                "pole_start_idx": pole_start_idx,
                "pole_start_date": pole_start_date,
                "pole_length": actual_pole_length,
                "pole_growth": pole_growth,
                "pole_start_price": pole_start_price,
                "max_price": max_price,
                "pole_height": pole_height,
                "flag_start_date": flag_start_date,
                "flag_length": flag_length,
                "flag_low": flag_low,
                "retracement": retracement,
                "breakout_idx": breakout_idx,
                "breakout_date": breakout_date,
                "breakout_price": breakout_price,
                "days_since_breakout": days_since_breakout,
                "target_price": target_price,
                "score": score
            })

    if not all_flags:
        return None

    # Wybieramy najlepszą formację (najwyższy score)
    best = max(all_flags, key=lambda x: x["score"])

    pole_start_str = best["pole_start_date"].strftime("%Y-%m-%d")
    flag_start_str = best["flag_start_date"].strftime("%Y-%m-%d")
    breakout_str = best["breakout_date"].strftime("%Y-%m-%d")

    return (
        f"🚩FLAG({best['days_since_breakout']}d ago): "
        f"Pole start:{pole_start_str}({best['pole_length']}d,+{best['pole_growth'] * 100:.1f}%), "
        f"Flag start:{flag_start_str}({best['flag_length']}d,ret{best['retracement'] * 100:.1f}%), "
        f"Breakout:{breakout_str}@{best['breakout_price']:.2f}, "
        f"MAX={best['max_price']:.2f}, target≈{best['target_price']:.2f}"
    )
