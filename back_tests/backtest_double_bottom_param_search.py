from __future__ import annotations

from time import perf_counter
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Optional

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


def max_gain_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> Optional[float]:
    future_window = df.iloc[event_idx + 1:event_idx + 1 + n_days]
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


def backtest_double_bottom_for_ticker(
        ticker: str,
        df: pd.DataFrame,
        params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if df.empty or len(df) < 40:
        return rows

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

    return rows


def build_param_sets() -> list[dict[str, Any]]:
    # Finalny, lokalny tuning wokół najlepszego obszaru: 3 * 5 * 2 * 2 = 60 kombinacji.
    pivot_left_values = [3]
    pivot_right_values = [3]
    min_days_between_bottoms_values = [10]
    max_days_between_bottoms_values = [120]
    max_bottom_price_diff_values = [0.03]
    min_neckline_rise_values = [0.06, 0.07, 0.08]
    min_neckline_rise_from_higher_bottom_values = [0.05]
    neckline_min_pos_ratio_values = [0.20]
    breakout_buffer_atr_values = [0.10]
    breakout_days_after_l2_ratio_values = [0.40, 0.45, 0.50, 0.55, 0.60]
    max_breakout_distance_above_neckline_values = [0.01]
    min_drop_into_l1_values = [0.05, 0.06]
    forbid_close_near_bottoms_max_above_values = [0.01, 0.012]
    require_downtrend_before_l1_values = [True]
    require_drop_into_l1_values = [True]
    forbid_any_pivot_low_between_values = [True]
    forbid_new_min_after_l2_before_breakout_values = [True]

    param_sets: list[dict[str, Any]] = []

    for (
            pivot_left,
            pivot_right,
            min_days_between_bottoms,
            max_days_between_bottoms,
            max_bottom_price_diff,
            min_neckline_rise,
            min_neckline_rise_from_higher_bottom,
            neckline_min_pos_ratio,
            breakout_buffer_atr,
            breakout_days_after_l2_ratio,
            max_breakout_distance_above_neckline,
            min_drop_into_l1,
            forbid_close_near_bottoms_max_above,
            require_downtrend_before_l1,
            require_drop_into_l1,
            forbid_any_pivot_low_between,
            forbid_new_min_after_l2_before_breakout,
    ) in product(
        pivot_left_values,
        pivot_right_values,
        min_days_between_bottoms_values,
        max_days_between_bottoms_values,
        max_bottom_price_diff_values,
        min_neckline_rise_values,
        min_neckline_rise_from_higher_bottom_values,
        neckline_min_pos_ratio_values,
        breakout_buffer_atr_values,
        breakout_days_after_l2_ratio_values,
        max_breakout_distance_above_neckline_values,
        min_drop_into_l1_values,
        forbid_close_near_bottoms_max_above_values,
        require_downtrend_before_l1_values,
        require_drop_into_l1_values,
        forbid_any_pivot_low_between_values,
        forbid_new_min_after_l2_before_breakout_values,
    ):
        param_sets.append(
            {
                "pivot_left": pivot_left,
                "pivot_right": pivot_right,
                "min_days_between_bottoms": min_days_between_bottoms,
                "max_days_between_bottoms": max_days_between_bottoms,
                "max_bottom_price_diff": max_bottom_price_diff,
                "min_neckline_rise": min_neckline_rise,
                "min_neckline_rise_from_higher_bottom": min_neckline_rise_from_higher_bottom,
                "neckline_min_pos_ratio": neckline_min_pos_ratio,
                "breakout_buffer_atr": breakout_buffer_atr,
                "atr_period": 14,
                "require_downtrend_before_l1": require_downtrend_before_l1,
                "downtrend_ma_period": 50,
                "require_drop_into_l1": require_drop_into_l1,
                "drop_into_l1_lookback_days": 30,
                "min_drop_into_l1": min_drop_into_l1,
                "max_l1_close_vs_recent_high": 0.97,
                "max_l1_close_vs_recent_avg": 0.98,
                "max_breakout_days_after_l2": 60,
                "breakout_days_after_l2_ratio": breakout_days_after_l2_ratio,
                "breakout_days_after_l2_min": 5,
                "breakout_days_after_l2_max": 40,
                "max_breakout_distance_above_neckline": max_breakout_distance_above_neckline,
                "forbid_close_below_bottoms_between": True,
                "forbid_close_below_bottoms_tolerance": 0.0,
                "forbid_close_near_bottoms_between": True,
                "forbid_close_near_bottoms_max_above": forbid_close_near_bottoms_max_above,
                "near_bottoms_exclude_days_after_l1": 3,
                "near_bottoms_exclude_days_before_l2": 3,
                "forbid_low_below_bottoms_between": True,
                "forbid_low_below_bottoms_tolerance": 0.0,
                "low_below_exclude_days_after_l1": 3,
                "low_below_exclude_days_before_l2": 3,
                "forbid_any_pivot_low_between": forbid_any_pivot_low_between,
                "pivot_low_between_exclude_days_after_l1": 3,
                "pivot_low_between_exclude_days_before_l2": 3,
                "forbid_new_min_after_l2_before_breakout": forbid_new_min_after_l2_before_breakout,
                "new_min_after_l2_exclude_days": 0,
                "new_min_after_l2_tolerance": 0.0,
            }
        )

    return param_sets


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"pivot={params['pivot_left']}/{params['pivot_right']}"
        f"_days={params['min_days_between_bottoms']}-{params['max_days_between_bottoms']}"
        f"_diff={params['max_bottom_price_diff']}"
        f"_neck={params['min_neckline_rise']}"
        f"_neck_h={params['min_neckline_rise_from_higher_bottom']}"
        f"_neck_pos={params['neckline_min_pos_ratio']}"
        f"_atr={params['breakout_buffer_atr']}"
        f"_bo_ratio={params['breakout_days_after_l2_ratio']}"
        f"_bo_ext={params['max_breakout_distance_above_neckline']}"
        f"_drop_l1={params['min_drop_into_l1']}"
        f"_mid_close={params['forbid_close_near_bottoms_max_above']}"
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
        "change_20d_pct",
        "max_gain_20d_pct",
        "max_gain_40d_pct",
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
        "change_20d_pct_avg",
        "max_gain_20d_pct_avg",
        "max_drawdown_5d_pct_avg",
    ]

    available_cols = [col for col in cols if col in summary_df.columns]

    print("TOP konfiguracje:")
    print(summary_df[available_cols].head(top_n).to_string(index=False))


def main() -> None:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 6 + 60)

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

            rows = backtest_double_bottom_for_ticker(
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
                        f"close={row['close_event']:.2f} | score={row['score']:.4f} | "
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
    summary_path = (
        f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/"
        f"double_bottom_param_search_summary_{timestamp}.csv"
    )
    details_path = (
        f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/"
        f"double_bottom_param_search_details_{timestamp}.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()