from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from data import ALL
from formations.flat_base import _check_flat_base_breakout_on_df


def download_history(company_abbr: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        df = yf.download(
            company_abbr + ".WA",
            start=start_date,
            end=end_date,
            auto_adjust=False,
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


def backtest_flat_base_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        touch_tolerance_pct: float = 0.005,
        min_touches_resistance: int = 3,
        min_breakout_pct: float = 0.01
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if df.empty or len(df) < 50:
        return rows

    previous_had_signal = False

    for event_idx in range(len(df)):
        history_until_event = df.iloc[:event_idx + 1].copy()

        signal = _check_flat_base_breakout_on_df(
            df=history_until_event,
            touch_tolerance_pct=touch_tolerance_pct,
            min_touches_resistance=min_touches_resistance,
            min_breakout_pct=min_breakout_pct,
        )

        has_signal = signal is not None

        if not has_signal:
            previous_had_signal = False
            continue

        if previous_had_signal:
            continue

        previous_had_signal = True

        rows.append({
            "ticker": ticker,
            "date": df.index[event_idx],
            "close_event": float(df.iloc[event_idx]["Close"]),
            "signal": signal,
            "change_3d_pct": close_change_after_n_days(df, event_idx, 3),
            "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
            "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
            "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
        })

    return rows


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("Brak breakoutów flat base.")
        return

    print(f"Liczba breakoutów: {len(results)}")
    print(f"Liczba spółek z breakoutami: {results['ticker'].nunique()}")
    print()

    metrics = ["change_3d_pct", "change_5d_pct", "change_10d_pct", "max_gain_20d_pct"]

    for col in metrics:
        series = results[col].dropna()
        if series.empty:
            print(f"{col}: brak danych")
            continue

        positive_count = int((series > 0).sum())
        negative_count = int((series < 0).sum())
        zero_count = int((series == 0).sum())

        print(
            f"{col}: "
            f"avg={series.mean():.4f}% | "
            f"median={series.median():.6f}% | "
            f"win_rate={(series > 0).mean() * 100:.1f}% | "
            f"pos={positive_count} | neg={negative_count} | zero={zero_count}"
        )

        around_zero = series.loc[series.abs().sort_values().index].head(10).tolist()
        print(f"  najbliżej zera: {[round(x, 6) for x in around_zero]}")
        print()

    print("Lista breakoutów:")
    for _, row in results.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        signal = row["signal"]
        close_event = row["close_event"]

        change_3d = row["change_3d_pct"]
        change_5d = row["change_5d_pct"]
        change_10d = row["change_10d_pct"]
        max_gain_20d = row["max_gain_20d_pct"]

        def fmt(value: float | None) -> str:
            if pd.isna(value):
                return "n/a"
            return f"{float(value):.2f}%"

        print(
            f"{row['ticker']} | {date_str} | close={close_event:.2f} | "
            f"3d={fmt(change_3d)} | 5d={fmt(change_5d)} | 10d={fmt(change_10d)} | "
            f"max20d={fmt(max_gain_20d)} | {signal}"
        )


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2 + 60)

    all_rows: list[dict[str, Any]] = []

    for ticker in ALL:
        print(f"Analiza: {ticker}")
        df = download_history(ticker, start_date, end_date)
        if df.empty:
            continue

        rows = backtest_flat_base_for_ticker(
            ticker=ticker,
            df=df,
            touch_tolerance_pct=0.005,
            min_touches_resistance=3,
            min_breakout_pct=0.01,
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("Brak wyników.")
        return

    results = results.sort_values(["ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/flat_base_backtest_results_{timestamp}.csv"
    results.to_csv(output_path, index=False)

    print()
    print_summary(results)
    print()
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    main()