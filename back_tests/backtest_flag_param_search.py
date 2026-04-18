from time import perf_counter
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from data import ALL
from formations.flag import find_flag_breakouts_on_df


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


def safe_pct_change(base_value: float, new_value: float) -> Optional[float]:
    if pd.isna(base_value) or pd.isna(new_value) or base_value == 0:
        return None
    return (float(new_value) / float(base_value) - 1.0) * 100.0


def close_change_after_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> Optional[float]:
    target_idx = event_idx + n_days
    if target_idx >= len(df):
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    target_close = float(df.iloc[target_idx]["Close"])
    return safe_pct_change(event_close, target_close)


def max_gain_next_20_days(df: pd.DataFrame, event_idx: int) -> Optional[float]:
    future_window = df.iloc[event_idx + 1:event_idx + 21]
    if future_window.empty:
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    max_high = float(future_window["High"].max())
    return safe_pct_change(event_close, max_high)


def max_drawdown_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> Optional[float]:
    future_window = df.iloc[event_idx + 1:event_idx + 1 + n_days]
    if future_window.empty:
        return None

    event_close = float(df.iloc[event_idx]["Close"])
    min_low = float(future_window["Low"].min())
    return safe_pct_change(event_close, min_low)


def backtest_flag_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    signals = find_flag_breakouts_on_df(
        df=df,
        pole_min_days=params["pole_min_days"],
        pole_max_days=params["pole_max_days"],
        pole_min_growth=params["pole_min_growth"],
        pole_max_daily_decline=params["pole_max_daily_decline"],
        max_days_without_new_high=params["max_days_without_new_high"],
        flag_min_days=params["flag_min_days"],
        flag_max_days_until_breakout=params["flag_max_days_until_breakout"],
        flag_max_retracement=params["flag_max_retracement"],
    )

    if not signals:
        return rows

    for signal_row in signals:
        event_date = pd.Timestamp(signal_row["date"])
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
                "signal": signal_row["signal"],
                "change_3d_pct": close_change_after_n_days(df, event_idx, 3),
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
                "max_drawdown_5d_pct": max_drawdown_next_n_days(df, event_idx, 5),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
            }
        )

    return rows


def build_param_sets() -> list[dict[str, Any]]:
    pole_min_days_values = [4]
    pole_max_days_values = [15]
    pole_min_growth_values = [0.06]
    pole_max_daily_decline_values = [0.33, 0.50]
    max_days_without_new_high_values = [2]
    flag_min_days_values = [3]
    flag_max_days_until_breakout_values = [20, 35]
    flag_max_retracement_values = [0.33, 0.50]

    param_sets: list[dict[str, Any]] = []

    for (
            pole_min_days,
            pole_max_days,
            pole_min_growth,
            pole_max_daily_decline,
            max_days_without_new_high,
            flag_min_days,
            flag_max_days_until_breakout,
            flag_max_retracement,
    ) in product(
        pole_min_days_values,
        pole_max_days_values,
        pole_min_growth_values,
        pole_max_daily_decline_values,
        max_days_without_new_high_values,
        flag_min_days_values,
        flag_max_days_until_breakout_values,
        flag_max_retracement_values,
    ):
        param_sets.append(
            {
                "pole_min_days": pole_min_days,
                "pole_max_days": pole_max_days,
                "pole_min_growth": pole_min_growth,
                "pole_max_daily_decline": pole_max_daily_decline,
                "max_days_without_new_high": max_days_without_new_high,
                "flag_min_days": flag_min_days,
                "flag_max_days_until_breakout": flag_max_days_until_breakout,
                "flag_max_retracement": flag_max_retracement,
            }
        )

    return param_sets


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"pole_min={params['pole_min_days']}"
        f"_pole_max={params['pole_max_days']}"
        f"_pole_growth={params['pole_min_growth']}"
        f"_pole_decline={params['pole_max_daily_decline']}"
        f"_no_high={params['max_days_without_new_high']}"
        f"_flag_min={params['flag_min_days']}"
        f"_flag_max={params['flag_max_days_until_breakout']}"
        f"_retr={params['flag_max_retracement']}"
    )


def summarize_results(results_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **params,
        "config": param_set_label(params),
        "signals_count": int(len(results_df)),
        "tickers_count": int(results_df["ticker"].nunique()) if not results_df.empty else 0,
    }

    metric_columns = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "max_gain_20d_pct",
        "max_drawdown_5d_pct",
        "max_drawdown_10d_pct",
    ]

    if results_df.empty:
        for col in metric_columns:
            summary[f"{col}_avg"] = None
            summary[f"{col}_median"] = None
            summary[f"{col}_positive_rate"] = None
        return summary

    for col in metric_columns:
        series = pd.to_numeric(results_df[col], errors="coerce").dropna()
        if series.empty:
            summary[f"{col}_avg"] = None
            summary[f"{col}_median"] = None
            summary[f"{col}_positive_rate"] = None
        else:
            summary[f"{col}_avg"] = float(series.mean())
            summary[f"{col}_median"] = float(series.median())
            summary[f"{col}_positive_rate"] = float((series > 0).mean() * 100.0)

    return summary


def print_top_configs(summary_df: pd.DataFrame, top_n: int = 10) -> None:
    if summary_df.empty:
        print("Brak wyników do wyświetlenia.")
        return

    cols = [
        "config",
        "signals_count",
        "tickers_count",
        "change_5d_pct_avg",
        "change_10d_pct_avg",
        "max_gain_20d_pct_avg",
        "max_drawdown_5d_pct_avg",
    ]

    available_cols = [col for col in cols if col in summary_df.columns]

    print("TOP konfiguracje:")
    print(summary_df[available_cols].head(top_n).to_string(index=False))


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5 + 60)

    print("Pobieranie danych tylko raz...")
    history_map: dict[str, pd.DataFrame] = {}

    for ticker in ALL:
        print(f"Pobieranie: {ticker}")
        df = download_history(ticker, start_date, end_date)
        if not df.empty:
            history_map[ticker] = df

    print()
    print(f"Pobrano dane dla {len(history_map)} tickerów.")
    print()

    param_sets = build_param_sets()
    all_summary_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []

    total_tickers = len(history_map)

    for idx, params in enumerate(param_sets, start=1):
        label = param_set_label(params)
        print(f"[{idx}/{len(param_sets)}] Test konfiguracji: {label}")

        config_start = perf_counter()
        config_rows: list[dict[str, Any]] = []

        for ticker_idx, (ticker, df) in enumerate(history_map.items(), start=1):
            ticker_start = perf_counter()

            rows = backtest_flag_for_ticker(
                ticker=ticker,
                df=df,
                params=params,
            )

            ticker_elapsed = perf_counter() - ticker_start
            config_elapsed = perf_counter() - config_start

            if rows:
                for row in rows:
                    print(
                        f"  {ticker} | breakout={pd.Timestamp(row['date']).strftime('%Y-%m-%d')} | "
                        f"close={row['close_event']:.2f} | signal={row['signal']} | "
                        f"ticker_time={ticker_elapsed:.2f}s | config_time={config_elapsed:.2f}s "
                        f"| progress={ticker_idx}/{total_tickers}"
                    )

            for row in rows:
                all_detail_rows.append(
                    {
                        "config": label,
                        **params,
                        **row,
                    }
                )

            config_rows.extend(rows)

        results_df = pd.DataFrame(config_rows)
        all_summary_rows.append(summarize_results(results_df, params))

        total_config_elapsed = perf_counter() - config_start
        print(
            f"  Zakończono konfigurację: {label} | "
            f"signals={len(config_rows)} | "
            f"time={total_config_elapsed:.2f}s"
        )
        print()

    summary_df = pd.DataFrame(all_summary_rows).sort_values(
        ["change_5d_pct_avg", "change_10d_pct_avg", "max_gain_20d_pct_avg"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flag_param_search_summary_{timestamp}.csv"
    details_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flag_param_search_details_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()