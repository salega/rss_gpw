from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from data import ALL
from formations.double_bottom import find_double_bottoms


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


def backtest_double_bottom_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if df.empty or len(df) < 40:
        return rows

    if params is None:
        params = {}

    prices = df.to_dict(orient="index")
    signals = find_double_bottoms(
        prices=prices,
        company_abbr=ticker,
        **params,
    )

    if not signals:
        return rows

    seen_breakout_dates: set[pd.Timestamp] = set()

    for signal_row in signals:
        event_date = pd.Timestamp(signal_row["breakout_date"])
        if event_date in seen_breakout_dates:
            continue
        seen_breakout_dates.add(event_date)

        event_idx_raw = signal_row.get("breakout_idx")
        if isinstance(event_idx_raw, int):
            event_idx = event_idx_raw
        else:
            try:
                event_idx = df.index.get_loc(event_date)
            except KeyError:
                continue

            if not isinstance(event_idx, int):
                continue

        rows.append(
            {
                "ticker": ticker,
                "date": event_date,
                "close_event": float(df.iloc[event_idx]["Close"]),
                "signal": (
                    f"🆆 {pd.Timestamp(signal_row['l1_date']).strftime('%Y-%m-%d')} / "
                    f"{pd.Timestamp(signal_row['l2_date']).strftime('%Y-%m-%d')} | "
                    f"neck={float(signal_row['neckline']):.2f} | "
                    f"target~{float(signal_row['target']):.2f}"
                ),
                "l1_date": pd.Timestamp(signal_row["l1_date"]),
                "l2_date": pd.Timestamp(signal_row["l2_date"]),
                "neckline_date": pd.Timestamp(signal_row["neckline_date"]),
                "breakout_date": event_date,
                "l1_price": float(signal_row["l1_price"]),
                "l2_price": float(signal_row["l2_price"]),
                "neckline": float(signal_row["neckline"]),
                "breakout_close": float(signal_row["breakout_close"]),
                "target": float(signal_row["target"]),
                "bottom_diff": float(signal_row["bottom_diff"]),
                "rise": float(signal_row["rise"]),
                "score": float(signal_row["score"]),
                "change_3d_pct": close_change_after_n_days(df, event_idx, 3),
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "change_20d_pct": close_change_after_n_days(df, event_idx, 20),
                "max_gain_20d_pct": max_gain_next_n_days(df, event_idx, 20),
                "max_gain_40d_pct": max_gain_next_n_days(df, event_idx, 40),
                "max_drawdown_5d_pct": max_drawdown_next_n_days(df, event_idx, 5),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
            }
        )

    rows.sort(key=lambda row: (row["date"], -row["score"]))
    return rows


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("Brak breakoutów podwójnego dna.")
        return

    print(f"Liczba breakoutów: {len(results)}")
    print(f"Liczba spółek z breakoutami: {results['ticker'].nunique()}")
    print()

    metrics = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "change_20d_pct",
        "max_gain_20d_pct",
        "max_gain_40d_pct",
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
            f"median={series.median():.4f}% | "
            f"win_rate={(series > 0).mean() * 100:.1f}% | "
            f"pos={positive_count} | neg={negative_count} | zero={zero_count}"
        )

    print()
    print("Lista breakoutów:")
    for _, row in results.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        l1_str = pd.Timestamp(row["l1_date"]).strftime("%Y-%m-%d")
        l2_str = pd.Timestamp(row["l2_date"]).strftime("%Y-%m-%d")

        def fmt(value: float | None) -> str:
            if pd.isna(value):
                return "n/a"
            return f"{float(value):.2f}%"

        print(
            f"{row['ticker']} | breakout={date_str} | "
            f"L1={l1_str} ({row['l1_price']:.2f}) | "
            f"L2={l2_str} ({row['l2_price']:.2f}) | "
            f"neck={row['neckline']:.2f} | target~{row['target']:.2f} | "
            f"score={row['score']:.4f} | "
            f"3d={fmt(row['change_3d_pct'])} | "
            f"5d={fmt(row['change_5d_pct'])} | "
            f"10d={fmt(row['change_10d_pct'])} | "
            f"20d={fmt(row['change_20d_pct'])}"
        )


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 6 + 60)

    params: dict[str, Any] = {
        "pivot_left": 3,
        "pivot_right": 3,
        "min_days_between_bottoms": 10,
        "max_days_between_bottoms": 120,
        "max_bottom_price_diff": 0.06,
        "min_neckline_rise": 0.06,
        "min_neckline_rise_from_higher_bottom": 0.06,
        "neckline_min_pos_ratio": 0.25,
        "breakout_buffer_atr": 0.25,
        "atr_period": 14,
        "require_downtrend_before_l1": True,
        "downtrend_ma_period": 50,
        "require_drop_into_l1": True,
        "drop_into_l1_lookback_days": 30,
        "min_drop_into_l1": 0.05,
        "max_l1_close_vs_recent_high": 0.97,
        "max_l1_close_vs_recent_avg": 0.98,
        "max_breakout_days_after_l2": 60,
        "breakout_days_after_l2_ratio": 0.6,
        "breakout_days_after_l2_min": 5,
        "breakout_days_after_l2_max": 40,
        "max_breakout_distance_above_neckline": 0.02,
        "forbid_close_below_bottoms_between": True,
        "forbid_close_below_bottoms_tolerance": 0.0,
        "forbid_close_near_bottoms_between": True,
        "forbid_close_near_bottoms_max_above": 0.01,
        "near_bottoms_exclude_days_after_l1": 3,
        "near_bottoms_exclude_days_before_l2": 3,
        "forbid_low_below_bottoms_between": True,
        "forbid_low_below_bottoms_tolerance": 0.0,
        "low_below_exclude_days_after_l1": 3,
        "low_below_exclude_days_before_l2": 3,
        "forbid_any_pivot_low_between": True,
        "pivot_low_between_exclude_days_after_l1": 3,
        "pivot_low_between_exclude_days_before_l2": 3,
        "forbid_new_min_after_l2_before_breakout": True,
        "new_min_after_l2_exclude_days": 0,
        "new_min_after_l2_tolerance": 0.0,
    }

    all_rows: list[dict[str, Any]] = []

    for ticker in ALL:
        print(f"Analiza: {ticker}")
        df = download_history(ticker, start_date, end_date)
        if df.empty:
            continue

        rows = backtest_double_bottom_for_ticker(
            ticker=ticker,
            df=df,
            params=params,
        )
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("Brak wyników.")
        return

    results = results.sort_values(["ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = (
        f"/Users/pl8000269/IdeaProjects/rss_gpw/"
        f"double_bottom_backtest_results_{timestamp}.csv"
    )
    results.to_csv(output_path, index=False)

    print()
    print_summary(results)
    print()
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    main()