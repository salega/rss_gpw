"""
Backtest parametryczny formacji odwróconej muszli zwyżkującej (Bulkowski Inverted Ascending Scallop).

Uruchomienie:
    python back_tests/backtest_scallop_param_search.py

Wyniki (CSV) trafiają do back_tests/reports/.
"""

from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from data import ALL, MARKET_SUFFIXES
from formations.scallop import find_scallop_signals


# ---------------------------------------------------------------------------
# Cache (identyczny mechanizm jak w pozostałych backtestach)
# ---------------------------------------------------------------------------

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


def download_history(
    company_abbr: str,
    start_date: datetime,
    end_date: datetime,
    market_suffix: str = ".WA",
) -> pd.DataFrame:
    ticker_symbol = company_abbr + market_suffix
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date,
                         auto_adjust=True, progress=False)
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
    market_suffix: str = ".WA",
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
        return downloaded_df.loc[
            (downloaded_df.index >= requested_start) & (downloaded_df.index <= requested_end)
        ]
    if not isinstance(cached_df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return cached_df.loc[
        (cached_df.index >= requested_start) & (cached_df.index <= requested_end)
    ].copy()


# ---------------------------------------------------------------------------
# Metryki transakcji
# ---------------------------------------------------------------------------

def safe_pct_change(base: float, new: float) -> float | None:
    if pd.isna(base) or pd.isna(new) or base == 0:
        return None
    return (float(new) / float(base) - 1.0) * 100.0


def trade_result(
    df: pd.DataFrame,
    event_idx: int,
    stop_price: float | None,
    drawdown_threshold: float = 0.20,
) -> dict:
    """
    Trzymaj pozycję do:
    1. Low < stop_price (B) → stop-loss
    2. Spadek ≥ 20% od szczytu → ultimate high (Bulkowski style)
    3. Koniec danych → wynik na ostatnim Close
    """
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return {"result_pct": None, "sl_hit": False, "peak_price": entry_close}

    peak_price = entry_close
    for i in range(len(future)):
        high = float(future.iloc[i]["High"])
        low = float(future.iloc[i]["Low"])

        if stop_price is not None and low < stop_price:
            return {"result_pct": safe_pct_change(entry_close, stop_price),
                    "sl_hit": True, "peak_price": peak_price}

        if high > peak_price:
            peak_price = high

        if peak_price > entry_close and (peak_price - low) / peak_price >= drawdown_threshold:
            return {"result_pct": safe_pct_change(entry_close, peak_price),
                    "sl_hit": False, "peak_price": peak_price}

    last_close = float(df.iloc[-1]["Close"])
    return {"result_pct": safe_pct_change(entry_close, last_close),
            "sl_hit": False, "peak_price": peak_price}


# ---------------------------------------------------------------------------
# Siatka parametrów
# ---------------------------------------------------------------------------

def build_param_sets() -> list[dict[str, Any]]:
    """
    Jedna bazowa konfiguracja wg Bulkowskiego.
    Bulkowski: avg retracement 53%, max 62%, median 17 dni do wybicia.
    """
    return [
        {
            "local_order": 7,
            "min_ac_rise_pct": 0.25,
            "min_ac_days": 15,
            "max_ac_days": 90,
            "min_retracement": 0.40,
            "max_retracement": 0.90,    # Bulkowski: unikaj 100%
            "max_breakout_days": 40,
            "min_arc_smoothness": 0.90,
            "max_rise_throwback": 0.12,  # max 12% cofnięcie w trakcie wzrostu A→C
            "require_uptrend_before_a": True,
            "uptrend_lookback": 40,
            "min_uptrend_pct": 0.15,
            "check_volume_decline": False,
        }
    ]


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"ord={params['local_order']}_"
        f"rise={params['min_ac_rise_pct']}_"
        f"ac={params['min_ac_days']}-{params['max_ac_days']}_"
        f"ret={params['min_retracement']}-{params['max_retracement']}_"
        f"brk={params['max_breakout_days']}"
    )


# ---------------------------------------------------------------------------
# Backtest jednego tickera
# ---------------------------------------------------------------------------

def backtest_scallop_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    params: dict[str, Any],
    signal_cutoff: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    min_bars = params["local_order"] * 2 + params["min_ac_days"] + 10
    if df.empty or len(df) < min_bars:
        return rows

    detection_df = df.loc[df.index <= signal_cutoff] if signal_cutoff is not None else df

    signals = find_scallop_signals(
        df=detection_df,
        local_order=params["local_order"],
        min_ac_rise_pct=params["min_ac_rise_pct"],
        min_ac_days=params["min_ac_days"],
        max_ac_days=params["max_ac_days"],
        min_retracement=params["min_retracement"],
        max_retracement=params["max_retracement"],
        max_breakout_days=params["max_breakout_days"],
        min_arc_smoothness=params.get("min_arc_smoothness", 0.85),
        max_rise_throwback=params.get("max_rise_throwback", 0.08),
        require_uptrend_before_a=params["require_uptrend_before_a"],
        uptrend_lookback=params["uptrend_lookback"],
        min_uptrend_pct=params["min_uptrend_pct"],
        check_volume_decline=params.get("check_volume_decline", False),
    )

    if not signals:
        return rows

    for sig in signals:
        event_date = pd.Timestamp(sig["date"])
        try:
            event_idx = df.index.get_loc(event_date)
        except KeyError:
            continue
        if isinstance(event_idx, slice):
            event_idx = event_idx.start
        elif hasattr(event_idx, "__index__"):
            event_idx = int(event_idx)
        if not isinstance(event_idx, int):
            continue

        # Stop = kilka centów poniżej B (Bulkowski: "stop below B")
        b_price = sig.get("b_price")
        stop_price = float(b_price) * 0.99 if b_price else None

        entry_price = float(df.iloc[event_idx]["Close"])
        tr = trade_result(df, event_idx, stop_price)

        rows.append({
            "ticker": ticker,
            "date": event_date,
            "close_event": entry_price,
            "signal": sig["signal"],
            "a_date": sig.get("a_date"),
            "c_date": sig.get("c_date"),
            "b_date": sig.get("b_date"),
            "a_price": sig.get("a_price"),
            "c_price": sig.get("c_price"),
            "b_price": b_price,
            "retracement_pct": sig.get("retracement_pct"),
            "ac_rise_pct": sig.get("ac_rise_pct"),
            "ac_days": sig.get("ac_days"),
            "bc_days": sig.get("bc_days"),
            "breakout_days": sig.get("breakout_days"),
            "stop_price": stop_price,
            "result_pct": tr["result_pct"],
            "sl_hit": tr["sl_hit"],
            "peak_reached": tr.get("peak_price"),
        })

    return rows


# ---------------------------------------------------------------------------
# Podsumowanie wyników
# ---------------------------------------------------------------------------

def summarize_results(results: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": param_set_label(params),
        **{k: params[k] for k in params},
        "trades": len(results),
        "tickers": int(results["ticker"].nunique()) if not results.empty else 0,
    }
    series = results["result_pct"].dropna() if (not results.empty and "result_pct" in results.columns) else pd.Series(dtype=float)
    summary["result_pct_count"] = int(series.shape[0])
    summary["result_pct_avg"] = float(series.mean()) if not series.empty else None
    summary["result_pct_median"] = float(series.median()) if not series.empty else None
    summary["result_pct_win_rate"] = float((series > 0).mean() * 100.0) if not series.empty else None

    if not results.empty and "sl_hit" in results.columns:
        sl = results["sl_hit"].dropna()
        summary["sl_hit_rate"] = float(sl.mean() * 100.0) if not sl.empty else None
        if "result_pct" in results.columns:
            successful = results.loc[results["sl_hit"] == False, "result_pct"].dropna()
            summary["result_pct_avg_no_sl"] = float(successful.mean()) if not successful.empty else None
            summary["result_pct_median_no_sl"] = float(successful.median()) if not successful.empty else None
            summary["result_pct_count_no_sl"] = int(successful.shape[0])
    else:
        summary["sl_hit_rate"] = None
        summary["result_pct_avg_no_sl"] = None
        summary["result_pct_median_no_sl"] = None
        summary["result_pct_count_no_sl"] = None

    return summary


def print_top_configs(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("Brak wyników.")
        return
    filtered = summary_df.loc[summary_df["trades"] >= 5].copy()
    if filtered.empty:
        filtered = summary_df.copy()
    cols = ["config", "trades", "tickers",
            "result_pct_avg", "result_pct_median", "result_pct_win_rate",
            "result_pct_count", "sl_hit_rate",
            "result_pct_avg_no_sl", "result_pct_median_no_sl"]
    available = [c for c in cols if c in filtered.columns]
    print("\nWyniki wg Bulkowskiego (ultimate high, stop = poniżej B):")
    print(filtered.sort_values("result_pct_avg", ascending=False)[available].head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def main() -> None:
    # ============================================================
    MARKET = "NYSE"
    TEST_TICKERS: list[str] | None = None  # None = wszystkie; ["BRKS"] = tylko ten ticker
    # ============================================================

    if MARKET == "GPW":
        all_tickers = ALL
        start_date = datetime(2013, 1, 1)
        end_date = datetime(2026, 5, 1)
        signal_cutoff = None
    elif MARKET == "NYSE":
        from data import ALL_US as all_tickers
        start_date = datetime(1985, 1, 1)
        end_date = datetime(2013, 1, 1)
        signal_cutoff = pd.Timestamp(datetime(2011, 1, 1))
    else:
        raise ValueError(f"Nieznany rynek: {MARKET}")

    tickers = TEST_TICKERS if TEST_TICKERS is not None else all_tickers
    market_suffix = MARKET_SUFFIXES.get(MARKET, ".WA")

    print(f"Rynek: {MARKET} | Okres sygnałów: {start_date.date()} → {signal_cutoff.date() if signal_cutoff else end_date.date()}")
    print("Ładowanie danych z cache / dociąganie braków...")
    history_map: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        print(f"Ładowanie: {ticker}")
        df = get_history_with_cache(ticker, start_date=start_date, end_date=end_date,
                                    market=MARKET, market_suffix=market_suffix)
        if not df.empty:
            history_map[ticker] = df

    print()
    print(f"Pobrano dane dla {len(history_map)} tickerów.")
    print()

    param_sets = build_param_sets()
    all_summary_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []

    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "n/a"
        return f"{float(v):.1f}%"

    for idx, params in enumerate(param_sets, start=1):
        label = param_set_label(params)
        print(f"[{idx}/{len(param_sets)}] Konfiguracja: {label}")
        print()

        config_rows: list[dict[str, Any]] = []

        for ticker, df in history_map.items():
            rows = backtest_scallop_for_ticker(
                ticker=ticker, df=df, params=params, signal_cutoff=signal_cutoff
            )
            for row in rows:
                a_date  = pd.Timestamp(row["a_date"]).strftime("%Y-%m-%d") if row.get("a_date") else "?"
                c_date  = pd.Timestamp(row["c_date"]).strftime("%Y-%m-%d") if row.get("c_date") else "?"
                b_date  = pd.Timestamp(row["b_date"]).strftime("%Y-%m-%d") if row.get("b_date") else "?"
                bo_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
                result  = row.get("result_pct")
                sl_str  = "🔴SL" if row.get("sl_hit") else "🟢ok"
                emoji   = "✅" if result is not None and result > 0 else "❌"
                print(
                    f"  🐚 {ticker}  ac={row.get('ac_days')}d  ret={fmt(row.get('retracement_pct'))}  rise={fmt(row.get('ac_rise_pct'))}\n"
                    f"     📈 A        {a_date}  @ {row.get('a_price', 0):.2f}\n"
                    f"     🔝 C (peak) {c_date}  @ {row.get('c_price', 0):.2f}\n"
                    f"     📉 B        {b_date}  @ {row.get('b_price', 0):.2f}\n"
                    f"     🚀 Breakout {bo_date}  @ {row.get('close_event', 0):.2f}\n"
                    f"     {sl_str} {emoji}  wynik={fmt(result)}\n"
                )
                all_detail_rows.append({"config": label, **params, **row})
            config_rows.extend(rows)

        print()
        summary = summarize_results(pd.DataFrame(config_rows), params)
        all_summary_rows.append(summary)
        print(
            f"[{idx}/{len(param_sets)}] Zakończono: {label} | "
            f"signals={len(config_rows)} | tickers={summary['tickers']}"
        )
        print()

    summary_df = pd.DataFrame(all_summary_rows).sort_values(
        ["result_pct_avg", "result_pct_median"], ascending=False, na_position="last"
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = "/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports"
    summary_path = f"{reports_dir}/scallop_param_search_summary_{MARKET}_{timestamp}.csv"
    details_path = f"{reports_dir}/scallop_param_search_details_{MARKET}_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)
    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()
