from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from data import ALL
from formations.flag import check_flag_breakout_today


def download_history(company_abbr: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        df = yf.download(
            company_abbr + ".WA",
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(company_abbr + ".WA", axis=1, level=-1)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    return df[["Open", "High", "Low", "Close"]].dropna().sort_index()


def safe_pct_change(base_value: float, new_value: float) -> float | None:
    if pd.isna(base_value) or pd.isna(new_value) or base_value == 0:
        return None
    return (float(new_value) / float(base_value) - 1.0) * 100.0


def close_change_after_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    target_idx = event_idx + n_days
    if target_idx >= len(df):
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    target_close = float(df.iloc[target_idx]["Close"])
    return safe_pct_change(event_close, target_close)


def max_gain_next_20_days(df: pd.DataFrame, event_idx: int) -> float | None:
    future_window = df.iloc[event_idx + 1:event_idx + 21]
    if future_window.empty:
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    max_high = float(future_window["High"].max())
    return safe_pct_change(event_close, max_high)


def max_drawdown_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    future_window = df.iloc[event_idx + 1:event_idx + 1 + n_days]
    if future_window.empty:
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    min_low = float(future_window["Low"].min())
    return safe_pct_change(event_close, min_low)


def gain_to_ultimate_high(df: pd.DataFrame, event_idx: int, drawdown_threshold: float = 0.20) -> float | None:
    """Bulkowski: zasięg od ceny wybicia do ostatecznego wierzchołka,
    po którym kurs spada o co najmniej drawdown_threshold (domyślnie 20%)."""
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return None

    peak_price = entry_close
    for i in range(len(future)):
        high = float(future.iloc[i]["High"])
        if high > peak_price:
            peak_price = high

        # sprawdź czy po tym szczycie nastąpił spadek ≥20%
        subsequent = future.iloc[i + 1:]
        if subsequent.empty:
            break
        min_subsequent_low = float(subsequent["Low"].min())
        if peak_price > 0 and (peak_price - min_subsequent_low) / peak_price >= drawdown_threshold:
            return safe_pct_change(entry_close, peak_price)

    # szczyt nie został potwierdzony (brak danych przyszłych) – zwróć None
    return None


def stop_loss_hit(df: pd.DataFrame, event_idx: int, flag_low: float) -> bool:
    """Bulkowski: stop = zamknięcie poniżej minimum flagi.
    Bulkowski wprost pisze: 'a closing price below the low posted in the flag'.
    Nie używamy intraday Low — tylko Close."""
    stop_price = flag_low
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return False
    return bool((future["Close"] < stop_price).any())


def backtest_flag_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        # Bulkowski WWF: maszt ≥90% w ≤40 sesjach
        pole_min_days: int = 4,
        pole_max_days: int = 40,
        pole_min_growth: float = 0.90,
        pole_max_daily_decline: float = 0.33,
        max_days_without_new_high: int = 2,
        # Bulkowski: flaga 5–19 dni
        flag_min_days: int = 5,
        flag_max_days_until_breakout: int = 19,
        require_volume_decline: bool = True,
        require_dense_flag: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    min_required_len = pole_min_days + flag_min_days + 1
    if df.empty or len(df) < min_required_len:
        return rows

    previous_had_signal = False

    for event_idx in range(len(df)):
        history_until_event = df.iloc[:event_idx + 1].copy()
        prices = history_until_event.to_dict(orient="index")

        signal = check_flag_breakout_today(
            prices=prices,
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

        has_signal = signal is not None

        if not has_signal:
            previous_had_signal = False
            continue

        if previous_had_signal:
            continue

        previous_had_signal = True

        event_close = float(df.iloc[event_idx]["Close"])

        # Pobierz flag_low z ostatniego sygnału (potrzebne do stop-lossa)
        # Szukamy flag_low w pełnych wynikach find_flag_breakouts_on_df
        from formations.flag import find_flag_breakouts_on_df
        full_results = find_flag_breakouts_on_df(
            df=history_until_event,
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
        flag_low = full_results[-1]["flag_low"] if full_results else None

        rows.append(
            {
                "ticker": ticker,
                "date": df.index[event_idx],
                "close_event": event_close,
                "signal": signal,
                # Bulkowski: zysk do ostatecznego wierzchołka (spadek ≥20%)
                "gain_to_ultimate_high_pct": gain_to_ultimate_high(df, event_idx, drawdown_threshold=0.20),
                # Pomocnicze okna (do porównań)
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "change_20d_pct": close_change_after_n_days(df, event_idx, 20),
                "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
                # Bulkowski: czy stop-loss (1 grosz pod minimum flagi) został trafiony?
                "stop_loss_hit": stop_loss_hit(df, event_idx, flag_low) if flag_low is not None else None,
                "flag_low": flag_low,
            }
        )

    return rows


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("Brak breakoutów flagi.")
        return

    print(f"Liczba breakoutów: {len(results)}")
    print(f"Liczba spółek z breakoutami: {results['ticker'].nunique()}")
    print()

    # Bulkowski: główna metryka – zysk do ostatecznego wierzchołka
    metrics = [
        "gain_to_ultimate_high_pct",  # Bulkowski: kluczowa metryka
        "change_5d_pct",
        "change_10d_pct",
        "change_20d_pct",
        "max_gain_20d_pct",
        "max_drawdown_10d_pct",
    ]

    for col in metrics:
        if col not in results.columns:
            continue
        series = results[col].dropna()
        if series.empty:
            print(f"{col}: brak danych")
            continue

        positive_count = int((series > 0).sum())
        negative_count = int((series < 0).sum())

        print(
            f"{col}: "
            f"avg={series.mean():.2f}% | "
            f"median={series.median():.2f}% | "
            f"win_rate={(series > 0).mean() * 100:.1f}% | "
            f"pos={positive_count} | neg={negative_count}"
        )
        print()

    # Bulkowski: stop-loss statystyki
    if "stop_loss_hit" in results.columns:
        sl = results["stop_loss_hit"].dropna()
        if not sl.empty:
            hit_rate = sl.mean() * 100
            print(f"Stop-loss (pod min. flagi) trafiony: {hit_rate:.1f}% transakcji")
            print()

    print("Lista breakoutów:")
    for _, row in results.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

        def fmt(value: Any) -> str:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return "n/a"
            return f"{float(value):.2f}%"

        sl_str = "SL:TAK" if row.get("stop_loss_hit") else "SL:NIE"
        print(
            f"{row['ticker']} | {date_str} | close={row['close_event']:.2f} | "
            f"ultimate={fmt(row.get('gain_to_ultimate_high_pct'))} | "
            f"5d={fmt(row.get('change_5d_pct'))} | "
            f"10d={fmt(row.get('change_10d_pct'))} | "
            f"20d={fmt(row.get('change_20d_pct'))} | "
            f"max20d={fmt(row.get('max_gain_20d_pct'))} | "
            f"dd10={fmt(row.get('max_drawdown_10d_pct'))} | "
            f"{sl_str} | "
            f"{row['signal']}"
        )


def main() -> None:
    end_date = datetime.today()
    # Bulkowski używał danych wieloletnich; bierzemy 5 lat żeby mieć szansę złapać ≥90% wzrosty
    start_date = end_date - timedelta(days=365 * 5)

    all_rows: list[dict[str, Any]] = []

    for ticker in ALL:
        print(f"Analiza: {ticker}")
        df = download_history(ticker, start_date, end_date)
        if df.empty:
            continue

        rows = backtest_flag_for_ticker(
            ticker=ticker,
            df=df,
            # Bulkowski WWF
            pole_min_days=4,
            pole_max_days=40,        # 2 miesiące sesyjne
            pole_min_growth=0.90,    # ≥90% wzrostu
            pole_max_daily_decline=0.33,
            max_days_without_new_high=2,
            flag_min_days=5,
            flag_max_days_until_breakout=19,  # Bulkowski: max 19 dni
            require_volume_decline=True,       # Bulkowski: wolumen maleje w fladze
            require_dense_flag=False,          # ustaw True dla wyższej jakości (53% vs 40%)
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("Brak wyników.")
        return

    results = results.sort_values(["ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flag_backtest_bulkowski_{timestamp}.csv"
    results.to_csv(output_path, index=False)

    print()
    print_summary(results)
    print()
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    main()