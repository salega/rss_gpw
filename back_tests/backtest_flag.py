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


def max_drawdown_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    future_window = df.iloc[event_idx + 1:event_idx + 1 + n_days]
    if future_window.empty:
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    min_low = float(future_window["Low"].min())
    return safe_pct_change(event_close, min_low)


def backtest_flag_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        pole_min_days: int = 3,
        pole_max_days: int = 20,
        pole_min_growth: float = 0.08,
        pole_max_daily_decline: float = 0.50,
        max_days_without_new_high: int = 2,
        flag_min_days: int = 3,
        flag_max_days_until_breakout: int = 35,
        flag_max_retracement: float = 0.50,
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
            flag_max_retracement=flag_max_retracement,
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
                "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
                "max_drawdown_5d_pct": max_drawdown_next_n_days(df, event_idx, 5),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
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

    metrics = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "max_gain_20d_pct",
        "max_drawdown_5d_pct",
        "max_drawdown_10d_pct",
    ]

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

        def fmt(value: float | None) -> str:
            if pd.isna(value):
                return "n/a"
            return f"{float(value):.2f}%"

        print(
            f"{row['ticker']} | {date_str} | close={row['close_event']:.2f} | "
            f"3d={fmt(row['change_3d_pct'])} | "
            f"5d={fmt(row['change_5d_pct'])} | "
            f"10d={fmt(row['change_10d_pct'])} | "
            f"max20d={fmt(row['max_gain_20d_pct'])} | "
            f"dd5={fmt(row['max_drawdown_5d_pct'])} | "
            f"dd10={fmt(row['max_drawdown_10d_pct'])} | "
            f"{row['signal']}"
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

        rows = backtest_flag_for_ticker(
            ticker=ticker,
            df=df,
            pole_min_days=3,
            pole_max_days=20,
            pole_min_growth=0.08,
            pole_max_daily_decline=0.50,
            max_days_without_new_high=2,
            flag_min_days=3,
            flag_max_days_until_breakout=35,
            flag_max_retracement=0.50,
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("Brak wyników.")
        return

    results = results.sort_values(["ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/flag_backtest_results_{timestamp}.csv"
    results.to_csv(output_path, index=False)

    print()
    print_summary(results)
    print()
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    main()