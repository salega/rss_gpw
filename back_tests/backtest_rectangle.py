from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from data import ALL
from formations.rectangle import _check_rectangle_breakout_on_df


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


def max_gain_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    future_window = df.iloc[event_idx + 1:event_idx + 1 + n_days]
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


def backtest_rectangle_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        touch_tolerance_of_height: float = 0.15,
        min_touches: int = 2,
        breakout_pct: float = 0.0,
        max_height_pct: float = 0.10,
        min_days_between_touches_ratio: float = 0.25,
        length_days_values: tuple[int, ...] = (20, 35, 50, 65, 80, 180),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if df.empty or len(df) < 50:
        return rows

    previous_had_signal = False

    for event_idx in range(len(df)):
        history_until_event = df.iloc[:event_idx + 1].copy()

        signal = _check_rectangle_breakout_on_df(
            df=history_until_event,
            touch_tolerance_of_height=touch_tolerance_of_height,
            min_touches=min_touches,
            breakout_pct=breakout_pct,
            max_height_pct=max_height_pct,
            min_days_between_touches_ratio=min_days_between_touches_ratio,
            length_days_values=length_days_values,
        )

        has_signal = signal is not None

        if not has_signal:
            previous_had_signal = False
            continue

        if previous_had_signal:
            continue

        previous_had_signal = True

        rows.append(
            {
                "ticker": ticker,
                "date": df.index[event_idx],
                "close_event": float(df.iloc[event_idx]["Close"]),
                "signal": signal,
                "change_3d_pct": close_change_after_n_days(df, event_idx, 3),
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "change_40d_pct": close_change_after_n_days(df, event_idx, 40),
                "max_gain_20d_pct": max_gain_next_n_days(df, event_idx, 20),
                "max_gain_40d_pct": max_gain_next_n_days(df, event_idx, 40),
                "max_drawdown_5d_pct": max_drawdown_next_n_days(df, event_idx, 5),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
            }
        )

    return rows


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("Brak sygnałów.")
        return

    metrics = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "change_40d_pct",
        "max_gain_20d_pct",
        "max_gain_40d_pct",
        "max_drawdown_5d_pct",
        "max_drawdown_10d_pct",
    ]

    print()
    print(f"Liczba sygnałów: {len(results)}")
    print(f"Liczba tickerów: {results['ticker'].nunique()}")

    for metric in metrics:
        series = results[metric].dropna()
        if series.empty:
            print(f"{metric}: brak danych")
            continue

        print(
            f"{metric}: avg={series.mean():.2f}% median={series.median():.2f}% "
            f"count={len(series)}"
        )


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5 + 60)

    all_rows: list[dict[str, Any]] = []

    for ticker in ALL:
        print(f"Pobieranie: {ticker}")
        df = download_history(ticker, start_date, end_date)
        if df.empty:
            continue

        rows = backtest_rectangle_for_ticker(
            ticker=ticker,
            df=df,
            touch_tolerance_of_height=0.10,
            min_touches=3,
            breakout_pct=0.01,
            max_height_pct=0.10,
            min_days_between_touches_ratio=0.25,
            length_days_values=(20, 35, 50),
        )
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print_summary(results_df)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/rectangle_backtest_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    main()