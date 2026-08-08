from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import product
from os import cpu_count
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


def build_param_sets() -> list[dict[str, Any]]:
    base_length_days_values = [20, 30, 40]
    min_breakout_values = [0.01, 0.025]
    max_center_deviation_values = [0.01, 0.02]
    use_low_for_depth_values = [True]

    param_sets: list[dict[str, Any]] = []

    for (
            base_length_days,
            min_breakout_pct,
            max_center_deviation_pct,
            use_low_for_depth,
    ) in product(
        base_length_days_values,
        min_breakout_values,
        max_center_deviation_values,
        use_low_for_depth_values,
    ):
        param_sets.append(
            {
                "base_length_days": base_length_days,
                "min_breakout_pct": min_breakout_pct,
                "max_center_deviation_pct": max_center_deviation_pct,
                "use_low_for_depth": use_low_for_depth,
            }
        )

    return param_sets


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"base={params['base_length_days']}_"
        f"breakout={params['min_breakout_pct']}_"
        f"center_dev={params['max_center_deviation_pct']}_"
        f"low={int(params['use_low_for_depth'])}"
    )


def backtest_flat_base_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if df.empty or len(df) < 50:
        return rows

    previous_had_signal = False

    for event_idx in range(len(df)):
        history_until_event = df.iloc[:event_idx + 1].copy()

        signal = _check_flat_base_breakout_on_df(
            df=history_until_event,
            min_breakout_pct=params["min_breakout_pct"],
            max_center_deviation_pct=params["max_center_deviation_pct"],
            use_low_for_depth=params["use_low_for_depth"],
            base_length_days_list=[params["base_length_days"]],
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
                "change_20d_pct": close_change_after_n_days(df, event_idx, 20),
                "change_50d_pct": close_change_after_n_days(df, event_idx, 50),
                "max_gain_10d_pct": max_gain_next_n_days(df, event_idx, 10),
                "max_gain_20d_pct": max_gain_next_n_days(df, event_idx, 20),
                "max_drawdown_5d_pct": max_drawdown_next_n_days(df, event_idx, 5),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
            }
        )

    return rows


def summarize_results(results: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": param_set_label(params),
        "base_length_days": params["base_length_days"],
        "min_breakout_pct": params["min_breakout_pct"],
        "max_center_deviation_pct": params["max_center_deviation_pct"],
        "use_low_for_depth": params["use_low_for_depth"],
        "trades": len(results),
        "tickers": int(results["ticker"].nunique()) if not results.empty else 0,
    }

    metrics = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "change_20d_pct",
        "change_50d_pct",
        "max_gain_10d_pct",
        "max_gain_20d_pct",
        "max_drawdown_5d_pct",
        "max_drawdown_10d_pct",
    ]

    for metric in metrics:
        series = results[metric].dropna() if not results.empty else pd.Series(dtype=float)

        summary[f"{metric}_count"] = int(series.shape[0])
        summary[f"{metric}_avg"] = float(series.mean()) if not series.empty else None
        summary[f"{metric}_median"] = float(series.median()) if not series.empty else None

        if "drawdown" in metric:
            summary[f"{metric}_win_rate"] = float((series > -3.0).mean() * 100.0) if not series.empty else None
        else:
            summary[f"{metric}_win_rate"] = float((series > 0).mean() * 100.0) if not series.empty else None

    return summary


def print_top_configs(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("Brak wyników do porównania.")
        return

    filtered = summary_df.loc[summary_df["trades"] >= 5].copy()
    if filtered.empty:
        filtered = summary_df.copy()

    cols = [
        "config",
        "trades",
        "tickers",
        "change_5d_pct_avg",
        "change_5d_pct_median",
        "change_5d_pct_win_rate",
        "change_10d_pct_avg",
        "change_10d_pct_median",
        "change_10d_pct_win_rate",
        "change_20d_pct_avg",
        "change_20d_pct_median",
        "change_20d_pct_win_rate",
        "change_50d_pct_avg",
        "change_50d_pct_median",
        "change_50d_pct_win_rate",
        "max_gain_10d_pct_avg",
        "max_gain_20d_pct_avg",
        "max_drawdown_5d_pct_avg",
        "max_drawdown_10d_pct_avg",
    ]

    print()
    print("TOP konfiguracje wg change_5d_pct_avg:")
    print(filtered.sort_values("change_5d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg change_10d_pct_avg:")
    print(filtered.sort_values("change_10d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg change_20d_pct_avg:")
    print(filtered.sort_values("change_20d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg change_50d_pct_avg:")
    print(filtered.sort_values("change_50d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg max_gain_10d_pct_avg:")
    print(filtered.sort_values("max_gain_10d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg max_gain_20d_pct_avg:")
    print(filtered.sort_values("max_gain_20d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg najmniejszego obsunięcia 5d:")
    print(filtered.sort_values("max_drawdown_5d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg najmniejszego obsunięcia 10d:")
    print(filtered.sort_values("max_drawdown_10d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))


def _run_single_config(
        args: tuple[dict[str, Any], dict[str, pd.DataFrame]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params, history_map = args
    label = param_set_label(params)

    config_rows: list[dict[str, Any]] = []
    config_detail_rows: list[dict[str, Any]] = []

    for ticker, df in history_map.items():
        rows = backtest_flat_base_for_ticker(
            ticker=ticker,
            df=df,
            params=params,
        )

        for row in rows:
            config_detail_rows.append(
                {
                    "config": label,
                    "base_length_days": params["base_length_days"],
                    "min_breakout_pct": params["min_breakout_pct"],
                    "max_center_deviation_pct": params["max_center_deviation_pct"],
                    "use_low_for_depth": params["use_low_for_depth"],
                    **row,
                }
            )

        config_rows.extend(rows)

    summary = summarize_results(pd.DataFrame(config_rows), params)
    return summary, config_detail_rows, config_rows


def main() -> None:
    # end_date = datetime.today()
    # start_date = end_date - timedelta(days=365 * 34 + 60)


    start_date = datetime(2013, 1, 1)
    end_date = datetime(2026, 5, 1)


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

    workers_limit = max(1, (cpu_count() or 2) - 1)
    max_workers = min(workers_limit, len(param_sets)) if param_sets else 1

    print(f"Uruchamianie równoległe dla konfiguracji: workers={max_workers}")
    print()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_config, (params, history_map)): (idx, params)
            for idx, params in enumerate(param_sets, start=1)
        }

        for future in as_completed(futures):
            idx, params = futures[future]
            label = param_set_label(params)

            try:
                summary, detail_rows, config_rows = future.result()
            except Exception as exc:
                print(f"[{idx}/{len(param_sets)}] Błąd konfiguracji {label}: {exc}")
                continue

            all_summary_rows.append(summary)
            all_detail_rows.extend(detail_rows)

            print(
                f"[{idx}/{len(param_sets)}] Zakończono: {label} | "
                f"signals={len(config_rows)} | tickers={summary['tickers']}"
            )

    summary_df = pd.DataFrame(all_summary_rows).sort_values(
        ["change_10d_pct_avg", "change_20d_pct_avg", "change_50d_pct_avg", "max_gain_10d_pct_avg", "max_gain_20d_pct_avg"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flat_base_param_search_summary_{timestamp}.csv"
    details_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flat_base_param_search_details_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()