from __future__ import annotations

from datetime import datetime, timedelta
from itertools import product
from typing import Any

import pandas as pd

from data import ALL
from back_tests.backtest_rectangle import (
    backtest_rectangle_for_ticker,
    download_history,
)


def build_param_sets() -> list[dict[str, Any]]:
    touch_tolerance_of_height_values = [0.15]
    min_touches_values = [2]
    breakout_pct_values = [0.02]
    max_height_pct_values = [0.15]
    min_days_between_touches_ratio_values = [0.15]
    length_days_values_list = [35, 50]

    param_sets: list[dict[str, Any]] = []

    for (
            touch_tolerance_of_height,
            min_touches,
            breakout_pct,
            max_height_pct,
            min_days_between_touches_ratio,
            length_days,
    ) in product(
        touch_tolerance_of_height_values,
        min_touches_values,
        breakout_pct_values,
        max_height_pct_values,
        min_days_between_touches_ratio_values,
        length_days_values_list,
    ):
        param_sets.append(
            {
                "touch_tolerance_of_height": touch_tolerance_of_height,
                "min_touches": min_touches,
                "breakout_pct": breakout_pct,
                "max_height_pct": max_height_pct,
                "min_days_between_touches_ratio": min_days_between_touches_ratio,
                "length_days_label": str(length_days),
                "length_days_values": (length_days,),
            }
        )

    return param_sets


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"touch_height={params['touch_tolerance_of_height']}_"
        f"min_touches={params['min_touches']}_"
        f"breakout={params['breakout_pct']}_"
        f"max_h={params['max_height_pct']}_"
        f"gap_ratio={params['min_days_between_touches_ratio']}_"
        f"lengths={params['length_days_label']}"
    )


def summarize_results(results: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": param_set_label(params),
        "touch_tolerance_of_height": params["touch_tolerance_of_height"],
        "min_touches": params["min_touches"],
        "breakout_pct": params["breakout_pct"],
        "max_height_pct": params["max_height_pct"],
        "min_days_between_touches_ratio": params["min_days_between_touches_ratio"],
        "length_days_label": params["length_days_label"],
        "length_days_values": ",".join(str(x) for x in params["length_days_values"]),
        "trades": len(results),
        "tickers": int(results["ticker"].nunique()) if not results.empty else 0,
    }

    metrics = [
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
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
        "touch_tolerance_of_height",
        "min_touches",
        "breakout_pct",
        "max_height_pct",
        "min_days_between_touches_ratio",
        "length_days_label",
        "change_5d_pct_avg",
        "change_5d_pct_median",
        "change_5d_pct_win_rate",
        "change_10d_pct_avg",
        "change_10d_pct_median",
        "change_10d_pct_win_rate",
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
    print("TOP konfiguracje wg max_gain_20d_pct_avg:")
    print(filtered.sort_values("max_gain_20d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg najmniejszego obsunięcia 5d:")
    print(filtered.sort_values("max_drawdown_5d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print()
    print("TOP konfiguracje wg najmniejszego obsunięcia 10d:")
    print(filtered.sort_values("max_drawdown_10d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))


def print_found_formations(details_df: pd.DataFrame) -> None:
    if details_df.empty:
        print()
        print("Nie znaleziono żadnych formacji.")
        return

    print()
    print("Znalezione formacje:")
    cols = [
        "config",
        "ticker",
        "date",
        "close_event",
        "signal",
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "max_gain_20d_pct",
        "max_drawdown_5d_pct",
        "max_drawdown_10d_pct",
    ]
    print(details_df[cols].to_string(index=False))


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

    for idx, params in enumerate(param_sets, start=1):
        label = param_set_label(params)
        print(f"[{idx}/{len(param_sets)}] Test konfiguracji: {label}")

        config_rows: list[dict[str, Any]] = []

        for ticker, df in history_map.items():
            rows = backtest_rectangle_for_ticker(
                ticker=ticker,
                df=df,
                touch_tolerance_of_height=params["touch_tolerance_of_height"],
                min_touches=params["min_touches"],
                breakout_pct=params["breakout_pct"],
                max_height_pct=params["max_height_pct"],
                min_days_between_touches_ratio=params["min_days_between_touches_ratio"],
                length_days_values=params["length_days_values"],
            )
            for row in rows:
                print(
                    f"FOUND | config={label} | ticker={row['ticker']} | date={row['date']} | "
                    f"close={row['close_event']} | signal={row['signal']}"
                )
                all_detail_rows.append(
                    {
                        "config": label,
                        "touch_tolerance_of_height": params["touch_tolerance_of_height"],
                        "min_touches": params["min_touches"],
                        "breakout_pct": params["breakout_pct"],
                        "max_height_pct": params["max_height_pct"],
                        "min_days_between_touches_ratio": params["min_days_between_touches_ratio"],
                        "length_days_label": params["length_days_label"],
                        "length_days_values": ",".join(str(x) for x in params["length_days_values"]),
                        **row,
                    }
                )
            config_rows.extend(rows)

        results_df = pd.DataFrame(config_rows)
        all_summary_rows.append(summarize_results(results_df, params))

    summary_df = pd.DataFrame(all_summary_rows).sort_values(
        ["change_5d_pct_avg", "change_10d_pct_avg", "max_gain_20d_pct_avg"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/rectangle_param_search_summary_{timestamp}.csv"
    details_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/rectangle_param_search_details_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_found_formations(details_df)
    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()