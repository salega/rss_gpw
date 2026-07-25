from pathlib import Path
from time import perf_counter
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from data import ALL_US, MARKET_SUFFIXES
from formations.flag import find_flag_breakouts_on_df


CACHE_DIR = Path("/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/cache")
CACHE_FILE_SUFFIX = ".pkl"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]
OPTIONAL_COLUMNS = ["Volume"]


def normalize_history_df(df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker_symbol, axis=1, level=-1)
        except (KeyError, IndexError):
            return pd.DataFrame()

    if "Date" in df.columns:
        df = df.set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~pd.isna(df.index)]

    if not set(REQUIRED_COLUMNS).issubset(set(df.columns)):
        return pd.DataFrame()

    cols_to_keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    normalized = df[cols_to_keep].dropna(subset=REQUIRED_COLUMNS).sort_index().copy()
    normalized.index = pd.DatetimeIndex(normalized.index).normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def download_history(company_abbr: str, start_date: datetime, end_date: datetime, market_suffix: str = "") -> pd.DataFrame:
    ticker_symbol = company_abbr + market_suffix

    try:
        df = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    return normalize_history_df(df, ticker_symbol)


def get_cache_file_path(market: str, ticker: str) -> Path:
    return CACHE_DIR / market / f"{ticker}{CACHE_FILE_SUFFIX}"


def load_cached_history(market: str, ticker: str) -> pd.DataFrame:
    cache_path = get_cache_file_path(market, ticker)
    if not cache_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_pickle(cache_path)
    except Exception:
        return pd.DataFrame()

    if "Date" in df.columns:
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    return normalize_history_df(df, ticker)


def save_cached_history(market: str, ticker: str, df: pd.DataFrame) -> None:
    cache_path = get_cache_file_path(market, ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)


def get_history_with_cache(
    company_abbr: str,
    start_date: datetime,
    end_date: datetime,
    market: str,
    market_suffix: str = "",
) -> pd.DataFrame:
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()

    cached_df = load_cached_history(market, company_abbr)

    if cached_df.empty:
        downloaded_df = download_history(
            company_abbr,
            requested_start.to_pydatetime(),
            (requested_end + pd.Timedelta(days=1)).to_pydatetime(),
            market_suffix=market_suffix,
        )
        if downloaded_df.empty or not isinstance(downloaded_df.index, pd.DatetimeIndex):
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        save_cached_history(market, company_abbr, downloaded_df)
        return downloaded_df.loc[(downloaded_df.index >= requested_start) & (downloaded_df.index <= requested_end)]

    if not isinstance(cached_df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return cached_df.loc[(cached_df.index >= requested_start) & (cached_df.index <= requested_end)]


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


def gain_to_ultimate_high(df: pd.DataFrame, event_idx: int, drawdown_threshold: float = 0.20) -> Optional[float]:
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

        subsequent = future.iloc[i + 1:]
        if subsequent.empty:
            break
        min_subsequent_low = float(subsequent["Low"].min())
        if peak_price > 0 and (peak_price - min_subsequent_low) / peak_price >= drawdown_threshold:
            return safe_pct_change(entry_close, peak_price)

    return None


def stop_loss_hit(df: pd.DataFrame, event_idx: int, flag_low: float) -> bool:
    """Bulkowski: stop 1 grosz poniżej minimum flagi."""
    stop_price = flag_low - 0.01
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return False
    return bool((future["Low"] < stop_price).any())


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
        require_volume_decline=params.get("require_volume_decline", True),
        require_dense_flag=params.get("require_dense_flag", False),
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

        flag_low = signal_row.get("flag_low")

        rows.append(
            {
                "ticker": ticker,
                "date": event_date,
                "close_event": float(df.iloc[event_idx]["Close"]),
                "signal": signal_row["signal"],
                "pole_growth_pct": signal_row.get("pole_growth_pct"),
                "retracement_pct": signal_row.get("retracement_pct"),
                "flag_days": signal_row.get("flag_days"),
                # Bulkowski: kluczowa metryka
                "gain_to_ultimate_high_pct": gain_to_ultimate_high(df, event_idx),
                # Pomocnicze
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "change_20d_pct": close_change_after_n_days(df, event_idx, 20),
                "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
                # Stop-loss
                "stop_loss_hit": stop_loss_hit(df, event_idx, flag_low) if flag_low is not None else None,
                "flag_low": flag_low,
            }
        )

    return rows


def build_param_sets() -> list[dict[str, Any]]:
    # Parametry bazowe wg Bulkowskiego — można rozszerzyć listy żeby przetestować warianty
    pole_min_days_values = [4]
    pole_max_days_values = [40]            # Bulkowski: 2 miesiące sesyjne
    pole_min_growth_values = [0.85]        # Bulkowski: "podwaja się lub prawie" — próg 85%
    pole_max_daily_decline_values = [0.20]  # nieużywane, zachowane dla kompatybilności
    max_days_without_new_high_values = [3]  # po ilu dniach bez nowego High maszt się kończy
    flag_min_days_values = [3]             # Bulkowski: min 3 dni (flaga może być krótka)
    flag_max_days_until_breakout_values = [19]  # Bulkowski: max 19 dni
    require_volume_decline_values = [False]   # bez filtra wolumenu — bazowa liczba formacji
    require_dense_flag_values = [False]       # bez filtra gęstości — bazowa liczba formacji

    param_sets: list[dict[str, Any]] = []

    for (
            pole_min_days,
            pole_max_days,
            pole_min_growth,
            pole_max_daily_decline,
            max_days_without_new_high,
            flag_min_days,
            flag_max_days_until_breakout,
            require_volume_decline,
            require_dense_flag,
    ) in product(
        pole_min_days_values,
        pole_max_days_values,
        pole_min_growth_values,
        pole_max_daily_decline_values,
        max_days_without_new_high_values,
        flag_min_days_values,
        flag_max_days_until_breakout_values,
        require_volume_decline_values,
        require_dense_flag_values,
    ):
        param_sets.append(
            {
                "pole_min_days": pole_min_days,
                "pole_max_days": pole_max_days,
                "pole_min_growth": pole_min_growth,
                "pole_max_daily_decline": pole_max_daily_decline,  # max korekta = X% wysokości masztu do tej pory
                "max_days_without_new_high": max_days_without_new_high,
                "flag_min_days": flag_min_days,
                "flag_max_days_until_breakout": flag_max_days_until_breakout,
                "require_volume_decline": require_volume_decline,
                "require_dense_flag": require_dense_flag,
            }
        )

    return param_sets


def param_set_label(params: dict[str, Any]) -> str:
    vol = "vol_dec" if params.get("require_volume_decline", True) else "vol_any"
    dense = "dense" if params.get("require_dense_flag", False) else "loose"
    return (
        f"pole_min={params['pole_min_days']}"
        f"_pole_max={params['pole_max_days']}"
        f"_pole_growth={params['pole_min_growth']}"
        f"_pole_decline={params['pole_max_daily_decline']}"
        f"_no_high={params['max_days_without_new_high']}"
        f"_flag_min={params['flag_min_days']}"
        f"_flag_max={params['flag_max_days_until_breakout']}"
        f"_{vol}_{dense}"
    )


def summarize_results(results_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **params,
        "config": param_set_label(params),
        "signals_count": int(len(results_df)),
        "tickers_count": int(results_df["ticker"].nunique()) if not results_df.empty else 0,
    }

    metric_columns = [
        "gain_to_ultimate_high_pct",   # Bulkowski: kluczowa metryka
        "change_5d_pct",
        "change_10d_pct",
        "change_20d_pct",
        "max_gain_20d_pct",
        "max_drawdown_10d_pct",
    ]

    if results_df.empty:
        for col in metric_columns:
            summary[f"{col}_avg"] = None
            summary[f"{col}_median"] = None
            summary[f"{col}_positive_rate"] = None
        summary["stop_loss_hit_rate"] = None
        return summary

    for col in metric_columns:
        if col not in results_df.columns:
            summary[f"{col}_avg"] = None
            summary[f"{col}_median"] = None
            summary[f"{col}_positive_rate"] = None
            continue
        series = pd.to_numeric(results_df[col], errors="coerce").dropna()
        if series.empty:
            summary[f"{col}_avg"] = None
            summary[f"{col}_median"] = None
            summary[f"{col}_positive_rate"] = None
        else:
            summary[f"{col}_avg"] = float(series.mean())
            summary[f"{col}_median"] = float(series.median())
            summary[f"{col}_positive_rate"] = float((series > 0).mean() * 100.0)

    # Bulkowski: procent transakcji z trafionym stop-lossem
    if "stop_loss_hit" in results_df.columns:
        sl_series = results_df["stop_loss_hit"].dropna()
        summary["stop_loss_hit_rate"] = float(sl_series.mean() * 100.0) if not sl_series.empty else None
    else:
        summary["stop_loss_hit_rate"] = None

    return summary


def print_top_configs(summary_df: pd.DataFrame, top_n: int = 10) -> None:
    if summary_df.empty:
        print("Brak wyników do wyświetlenia.")
        return

    cols = [
        "config",
        "signals_count",
        "tickers_count",
        "gain_to_ultimate_high_pct_avg",   # Bulkowski: kluczowa
        "gain_to_ultimate_high_pct_median",
        "change_10d_pct_avg",
        "max_gain_20d_pct_avg",
        "max_drawdown_10d_pct_avg",
        "stop_loss_hit_rate",
    ]

    available_cols = [col for col in cols if col in summary_df.columns]

    print("TOP konfiguracje (sortowane wg gain_to_ultimate_high):")
    print(summary_df[available_cols].head(top_n).to_string(index=False))


def main() -> None:
    start_date = datetime(1985, 1, 1)  # Bulkowski: od stycznia 1985
    end_date = datetime(2011, 1, 1)    # Bulkowski: do stycznia 2011
    market = "NYSE"
    market_suffix = MARKET_SUFFIXES[market]

    print("Ładowanie danych z cache / dociąganie braków...")
    history_map: dict[str, pd.DataFrame] = {}

    for ticker in ALL_US[:1]:
    # for ticker in ALL_US:
        print(f"Ładowanie: {ticker}")
        df = get_history_with_cache(
            ticker,
            start_date=start_date,
            end_date=end_date,
            market=market,
            market_suffix=market_suffix,
        )
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
        ["gain_to_ultimate_high_pct_avg", "max_gain_20d_pct_avg", "change_10d_pct_avg"],
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