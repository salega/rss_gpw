"""
Backtest parametryczny formacji podwójnego dna (Bulkowski Eve & Eve / Adam & Adam …)
Analogiczny do backtest_flat_base_param_search.py.

Uruchomienie:
    python back_tests/backtest_double_bottom_param_search.py

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
from formations.double_bottom import find_double_bottom_signals


# ---------------------------------------------------------------------------
# Cache (identyczny mechanizm jak w backtest_flag_param_search.py)
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
        df = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
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
# Metryki
# ---------------------------------------------------------------------------

def safe_pct_change(base_value: float, new_value: float) -> float | None:
    if pd.isna(base_value) or pd.isna(new_value) or base_value == 0:
        return None
    return (float(new_value) / float(base_value) - 1.0) * 100.0


def close_change_after_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    target_idx = event_idx + n_days
    if target_idx >= len(df):
        return None
    return safe_pct_change(float(df.iloc[event_idx]["Close"]), float(df.iloc[target_idx]["Close"]))


def max_gain_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    window = df.iloc[event_idx + 1: event_idx + 1 + n_days]
    if window.empty:
        return None
    return safe_pct_change(float(df.iloc[event_idx]["Close"]), float(window["High"].max()))


def max_drawdown_next_n_days(df: pd.DataFrame, event_idx: int, n_days: int) -> float | None:
    window = df.iloc[event_idx + 1: event_idx + 1 + n_days]
    if window.empty:
        return None
    return safe_pct_change(float(df.iloc[event_idx]["Close"]), float(window["Low"].min()))


# ---------------------------------------------------------------------------
# Siatka parametrów
# ---------------------------------------------------------------------------

def build_param_sets() -> list[dict[str, Any]]:
    """
    Jedna konfiguracja bazowa wg Bulkowskiego (Eve & Eve identification guidelines):
    - local_min_order=5      : okno 5 sesji po każdej stronie dla wykrycia lokalnego dołka
    - min_separation_days=10 : min ~2 tygodnie między dołkami
    - max_separation_days=105: max ~5 miesięcy (Bulkowski: avg 2 miesiące, dopuszcza więcej)
    - max_bottom_diff_pct=0.06: max 6% różnicy cen dołków (Bulkowski: avg 2%, toleruje do ~6%)
    - min_peak_rise_pct=0.10 : min 10% wzrost między dołkami (Bulkowski: avg 26%, min 10%)
    - require_downtrend=True : formacja musi być poprzedzona trendem spadkowym
    - check_volume=True      : wyższy wolumen na lewym dnie (Bulkowski: "usually higher on left")
    """
    return [
        {
            "local_min_order": 5,
            "min_separation_days": 10,
            "max_separation_days": 70,
            "max_bottom_diff_pct": 0.03,   # Bulkowski avg 2%, toleruje do ~3%
            "min_peak_rise_pct": 0.19,     # Bulkowski median 19%
            "require_downtrend": True,
            "check_volume": False,
        }
    ]


def param_set_label(params: dict[str, Any]) -> str:
    return (
        f"ord={params['local_min_order']}_"
        f"sep={params['min_separation_days']}-{params['max_separation_days']}_"
        f"diff={params['max_bottom_diff_pct']}_"
        f"rise={params['min_peak_rise_pct']}_"
        f"down={int(params['require_downtrend'])}_"
        f"vol={int(params['check_volume'])}"
    )


# ---------------------------------------------------------------------------
# Backtest jednego tickera
# ---------------------------------------------------------------------------

def backtest_double_bottom_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    min_bars = params["local_min_order"] * 2 + params["min_separation_days"] + 20
    if df.empty or len(df) < min_bars:
        return rows

    # Skanuj cały df jednorazowo — O(n), nie O(n²)
    signals = find_double_bottom_signals(
        df=df,
        local_min_order=params["local_min_order"],
        min_separation_days=params["min_separation_days"],
        max_separation_days=params["max_separation_days"],
        max_bottom_diff_pct=params["max_bottom_diff_pct"],
        min_peak_rise_pct=params["min_peak_rise_pct"],
        require_downtrend=params["require_downtrend"],
        check_volume=params["check_volume"],
    )

    if not signals:
        return rows

    for sig in signals:
        event_date = pd.Timestamp(sig["date"])
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
                "signal": sig["signal"],
                "pattern_type": sig.get("pattern_type"),
                "left_trough_date": sig.get("left_trough_date"),
                "right_trough_date": sig.get("right_trough_date"),
                "peak_date": sig.get("peak_date"),
                "left_trough_price": sig.get("left_trough_price"),
                "right_trough_price": sig.get("right_trough_price"),
                "peak_price": sig.get("peak_price"),
                "separation_days": sig.get("separation_days"),
                "peak_rise_pct": sig.get("peak_rise_pct"),
                "bottom_diff_pct": sig.get("bottom_diff_pct"),
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


# ---------------------------------------------------------------------------
# Wyświetlanie rankingu konfiguracji
# ---------------------------------------------------------------------------

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

    for sort_col in [
        "change_5d_pct_avg",
        "change_10d_pct_avg",
        "change_20d_pct_avg",
        "change_50d_pct_avg",
        "max_gain_10d_pct_avg",
        "max_gain_20d_pct_avg",
    ]:
        print(f"\nTOP konfiguracje wg {sort_col}:")
        print(filtered.sort_values(sort_col, ascending=False)[cols].head(15).to_string(index=False))

    print("\nTOP konfiguracje wg najmniejszego obsunięcia 5d:")
    print(filtered.sort_values("max_drawdown_5d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))

    print("\nTOP konfiguracje wg najmniejszego obsunięcia 10d:")
    print(filtered.sort_values("max_drawdown_10d_pct_avg", ascending=False)[cols].head(15).to_string(index=False))


# ---------------------------------------------------------------------------
# Równoległa pętla po konfiguracjach
# ---------------------------------------------------------------------------

def _run_single_config(
    args: tuple[dict[str, Any], dict[str, pd.DataFrame]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params, history_map = args
    label = param_set_label(params)

    config_rows: list[dict[str, Any]] = []
    config_detail_rows: list[dict[str, Any]] = []

    for ticker, df in history_map.items():
        rows = backtest_double_bottom_for_ticker(ticker=ticker, df=df, params=params)
        for row in rows:
            config_detail_rows.append({"config": label, **params, **row})
        config_rows.extend(rows)

    summary = summarize_results(pd.DataFrame(config_rows), params)
    return summary, config_detail_rows, config_rows


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def main() -> None:
    # ============================================================
    # KONFIGURACJA
    MARKET = "NYSE"          # "GPW" lub "NYSE"
    TEST_TICKERS: list[str] | None = None  # None = wszystkie; ["LLY"] = tylko ten ticker
    # ============================================================

    if MARKET == "GPW":
        all_tickers = ALL
        start_date = datetime(2013, 1, 1)
        end_date = datetime(2026, 5, 1)
    elif MARKET == "NYSE":
        from data import ALL_US as all_tickers
        start_date = datetime(1985, 1, 1)
        end_date = datetime(2011, 1, 1)   # Bulkowski: styczeń 1985 – styczeń 2011
    else:
        raise ValueError(f"Nieznany rynek: {MARKET}")

    tickers = TEST_TICKERS if TEST_TICKERS is not None else all_tickers
    market_suffix = MARKET_SUFFIXES.get(MARKET, ".WA")

    print(f"Rynek: {MARKET} | Okres: {start_date.date()} → {end_date.date()}")
    print("Ładowanie danych z cache / dociąganie braków...")
    history_map: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        print(f"Ładowanie: {ticker}")
        df = get_history_with_cache(
            ticker,
            start_date=start_date,
            end_date=end_date,
            market=MARKET,
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
            rows = backtest_double_bottom_for_ticker(ticker=ticker, df=df, params=params)

            for row in rows:
                l1_date  = pd.Timestamp(row["left_trough_date"]).strftime("%Y-%m-%d")
                l2_date  = pd.Timestamp(row["right_trough_date"]).strftime("%Y-%m-%d")
                nk_date  = pd.Timestamp(row["peak_date"]).strftime("%Y-%m-%d")
                bo_date  = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
                l1_price = row["left_trough_price"]
                l2_price = row["right_trough_price"]
                nk_price = row["peak_price"]
                bo_price = row["close_event"]
                ptype    = row.get("pattern_type", "?")
                sep      = row.get("separation_days", "?")
                print(
                    f"  📈 {ticker}  [{ptype}]  sep={sep}d\n"
                    f"     📉 L1      {l1_date}  @ {l1_price:.2f}\n"
                    f"     🔝 Neck    {nk_date}  @ {nk_price:.2f}  (rise={fmt(row.get('peak_rise_pct'))})\n"
                    f"     📉 L2      {l2_date}  @ {l2_price:.2f}  (diff={fmt(row.get('bottom_diff_pct'))})\n"
                    f"     🚀 Breakout {bo_date}  @ {bo_price:.2f}\n"
                    f"     📊 Wyniki:  5d={fmt(row.get('change_5d_pct'))}  "
                    f"10d={fmt(row.get('change_10d_pct'))}  "
                    f"20d={fmt(row.get('change_20d_pct'))}  "
                    f"50d={fmt(row.get('change_50d_pct'))}  "
                    f"max20d={fmt(row.get('max_gain_20d_pct'))}  "
                    f"dd10={fmt(row.get('max_drawdown_10d_pct'))}\n"
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
        ["change_10d_pct_avg", "change_20d_pct_avg", "change_50d_pct_avg",
         "max_gain_10d_pct_avg", "max_gain_20d_pct_avg"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = "/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports"
    summary_path = f"{reports_dir}/double_bottom_param_search_summary_{MARKET}_{timestamp}.csv"
    details_path = f"{reports_dir}/double_bottom_param_search_details_{MARKET}_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()
