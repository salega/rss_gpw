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


def trade_result_bulkowski_stop(df: pd.DataFrame, event_idx: int, flag_low: Optional[float],
                                atr_multiplier: float = 2.0, atr_period: int = 14,
                                weakening_days: int = 3) -> dict:
    """
    Strategia wyjścia "Bulkowski-style" — wychodzi gdy spełnione są 2 z 3 warunków:
    1. Stop zmienności: Close < szczyt - atr_multiplier * ATR (ruchomy, tylko w górę)
    2. Słabnięcie kursu: Close < Close sprzed weakening_days dni
    3. Brak nowego szczytu od weakening_days dni z rzędu

    Przy 1 warunku trzymasz. Przy 2+ wychodzisz na następnym Close.
    flag_low chroni jako bezwarunkowy stop na początku.
    """
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}

    initial_atr = calc_atr(df, event_idx, period=atr_period)
    initial_stop = entry_close - atr_multiplier * initial_atr
    if flag_low is not None:
        initial_stop = max(initial_stop, flag_low)

    current_stop = initial_stop
    peak_price = entry_close
    days_without_new_high = 0
    closes: list[float] = [entry_close]

    for i in range(len(future)):
        future_idx = event_idx + 1 + i
        close = float(future.iloc[i]["Close"])
        atr = calc_atr(df, future_idx, period=atr_period)

        # bezwarunkowy stop: flag_low
        if flag_low is not None and close < flag_low:
            result = safe_pct_change(entry_close, close)
            return {"result_pct": result, "sl_hit": True, "exit_price": close}

        # aktualizuj szczyt i dni bez nowego szczytu
        if close > peak_price:
            peak_price = close
            days_without_new_high = 0
        else:
            days_without_new_high += 1

        # przesuń stop zmienności w górę (nigdy w dół)
        new_stop = close - atr_multiplier * atr
        current_stop = max(current_stop, new_stop)

        closes.append(close)

        # sprawdź warunki wyjścia
        cond1_atr_stop = close < current_stop
        cond2_weakening = len(closes) > weakening_days and close < closes[-(weakening_days + 1)]
        cond3_no_new_high = days_without_new_high >= weakening_days

        signals_triggered = sum([cond1_atr_stop, cond2_weakening, cond3_no_new_high])

        if signals_triggered >= 2:
            result = safe_pct_change(entry_close, close)
            return {"result_pct": result, "sl_hit": False, "exit_price": close}

    return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}


def calc_atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    """ATR (Average True Range) — mierzy dzienną zmienność kursu."""
    start = max(0, idx - period)
    window = df.iloc[start:idx + 1]
    if len(window) < 2:
        return float(df.iloc[idx]["High"]) - float(df.iloc[idx]["Low"])
    true_ranges = []
    for i in range(1, len(window)):
        high = float(window.iloc[i]["High"])
        low = float(window.iloc[i]["Low"])
        prev_close = float(window.iloc[i - 1]["Close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges)


def trade_result_trailing_atr(df: pd.DataFrame, event_idx: int, flag_low: Optional[float], atr_multiplier: float = 2.0, atr_period: int = 14, min_gain_to_activate: float = 0.0) -> dict:
    """
    Trailing stop oparty o ATR: stop = szczyt - atr_multiplier * ATR(atr_period).
    Aktywny od pierwszego dnia (min_gain_to_activate=0.0 = bez progu aktywacji).
    flag_low ma zawsze priorytet.
    """
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}

    peak_price = entry_close
    trailing_active = False

    for i in range(len(future)):
        future_idx = event_idx + 1 + i
        high = float(future.iloc[i]["High"])
        low = float(future.iloc[i]["Low"])

        if flag_low is not None and low < flag_low:
            result = safe_pct_change(entry_close, flag_low)
            return {"result_pct": result, "sl_hit": True, "exit_price": flag_low}

        if high > peak_price:
            peak_price = high

        if not trailing_active and peak_price >= entry_close * (1 + min_gain_to_activate):
            trailing_active = True

        if trailing_active:
            atr = calc_atr(df, future_idx, period=atr_period)
            trailing_stop_price = peak_price - atr_multiplier * atr
            if low < trailing_stop_price:
                result = safe_pct_change(entry_close, trailing_stop_price)
                return {"result_pct": result, "sl_hit": False, "exit_price": trailing_stop_price}

    return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}


def trade_result_trailing_stop(df: pd.DataFrame, event_idx: int, flag_low: Optional[float], trailing_pct: float = 0.10, min_gain_to_activate: float = 0.15, initial_stop_pct: float = 0.10) -> dict:
    """
    Trailing stop z initial stop:
    1. Przed aktywacją trailing stopu: initial stop = entry * (1 - initial_stop_pct)
       (Bulkowski używał ~10% od ceny wejścia jako początkowy stop)
    2. Po osiągnięciu +min_gain_to_activate: trailing stop = szczyt * (1 - trailing_pct)
    3. flag_low zawsze ma priorytet (jeśli jest niżej niż initial stop)

    Zwraca dict z:
      - result_pct: wynik transakcji w %
      - sl_hit: True jeśli stop-loss trafiony
      - exit_price: cena wyjścia
    """
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}

    peak_price = entry_close
    trailing_active = False
    initial_stop_price = entry_close * (1 - initial_stop_pct)
    # initial stop nie może być wyżej niż flag_low
    effective_initial_stop = max(initial_stop_price, flag_low) if flag_low is not None else initial_stop_price

    for i in range(len(future)):
        high = float(future.iloc[i]["High"])
        low = float(future.iloc[i]["Low"])

        if not trailing_active:
            # przed aktywacją trailing stopu: initial stop chroni pozycję
            if low < effective_initial_stop:
                result = safe_pct_change(entry_close, effective_initial_stop)
                return {"result_pct": result, "sl_hit": True, "exit_price": effective_initial_stop}
        else:
            # po aktywacji: flag_low nadal ma priorytet
            if flag_low is not None and low < flag_low:
                result = safe_pct_change(entry_close, flag_low)
                return {"result_pct": result, "sl_hit": True, "exit_price": flag_low}

        # aktualizuj szczyt
        if high > peak_price:
            peak_price = high

        # aktywuj trailing stop po +min_gain_to_activate
        if not trailing_active and peak_price >= entry_close * (1 + min_gain_to_activate):
            trailing_active = True

        if trailing_active:
            trailing_stop_price = peak_price * (1 - trailing_pct)
            if low < trailing_stop_price:
                result = safe_pct_change(entry_close, trailing_stop_price)
                return {"result_pct": result, "sl_hit": False, "exit_price": trailing_stop_price}

    return {"result_pct": None, "sl_hit": False, "exit_price": entry_close}


def trade_result(df: pd.DataFrame, event_idx: int, flag_low: Optional[float], drawdown_threshold: float = 0.20) -> dict:
    """
    Symuluje transakcję po wybiciu. Dwa scenariusze (co nastąpi pierwsze):
    1. Low spada poniżej flag_low → ❌ stop-loss, nieudana transakcja
    2. Low spada ≥20% od bieżącego szczytu → ✅ wychodzisz na szczycie

    Zwraca dict z:
      - result_pct: wynik transakcji w %
      - sl_hit: True jeśli stop-loss
      - peak_price: szczyt osiągnięty przed zamknięciem
    """
    entry_close = float(df.iloc[event_idx]["Close"])
    future = df.iloc[event_idx + 1:]
    if future.empty:
        return {"result_pct": None, "sl_hit": False, "peak_price": entry_close}

    peak_price = entry_close

    for i in range(len(future)):
        high = float(future.iloc[i]["High"])
        low = float(future.iloc[i]["Low"])

        # najpierw sprawdź stop-loss (Low < flag_low) — priorytet przed szczytem
        if flag_low is not None and low < flag_low:
            # stop-loss trafiony — wynik to zysk/strata do tego momentu (cena = flag_low)
            result = safe_pct_change(entry_close, flag_low)
            return {"result_pct": result, "sl_hit": True, "peak_price": peak_price}

        # aktualizuj szczyt
        if high > peak_price:
            peak_price = high

        # szczyt potwierdzony spadkiem ≥20%
        if peak_price > 0 and (peak_price - low) / peak_price >= drawdown_threshold:
            result = safe_pct_change(entry_close, peak_price)
            return {"result_pct": result, "sl_hit": False, "peak_price": peak_price}

    return {"result_pct": None, "sl_hit": False, "peak_price": peak_price}


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
        entry_price = float(df.iloc[event_idx]["Close"])

        tr = trade_result(df, event_idx, flag_low)
        actual_result_pct = tr["result_pct"]
        sl_hit = tr["sl_hit"]

        # trailing 20% od szczytu, initial stop 20% od ceny wejścia
        tr10 = trade_result_trailing_stop(df, event_idx, flag_low, trailing_pct=0.20, min_gain_to_activate=0.0, initial_stop_pct=0.20)
        trailing_10pct_result = tr10["result_pct"]

        tr_atr = trade_result_trailing_atr(df, event_idx, flag_low, atr_multiplier=2.0, min_gain_to_activate=0.0)
        trailing_atr_result = tr_atr["result_pct"]

        tr_bul = trade_result_bulkowski_stop(df, event_idx, flag_low, atr_multiplier=2.0)
        bulkowski_stop_result = tr_bul["result_pct"]

        rows.append(
            {
                "ticker": ticker,
                "date": event_date,
                "close_event": entry_price,
                "signal": signal_row["signal"],
                "pole_growth_pct": signal_row.get("pole_growth_pct"),
                "retracement_pct": signal_row.get("retracement_pct"),
                "flag_days": signal_row.get("flag_days"),
                "actual_result_pct": actual_result_pct,
                "trailing_10pct_result": trailing_10pct_result,  # trailing stop 10% od szczytu
                "trailing_atr_result": trailing_atr_result,      # trailing stop 2*ATR (Bulkowski)
                "bulkowski_stop_result": bulkowski_stop_result,  # stop zmienności jak Bulkowski (Close - 2*ATR, tylko w górę)
                "change_5d_pct": close_change_after_n_days(df, event_idx, 5),
                "change_10d_pct": close_change_after_n_days(df, event_idx, 10),
                "change_20d_pct": close_change_after_n_days(df, event_idx, 20),
                "max_gain_20d_pct": max_gain_next_20_days(df, event_idx),
                "max_drawdown_10d_pct": max_drawdown_next_n_days(df, event_idx, 10),
                "stop_loss_hit": sl_hit,
                "flag_low": flag_low,
            }
        )

    return rows


def build_param_sets() -> list[dict[str, Any]]:
    # Parametry bazowe wg Bulkowskiego — można rozszerzyć listy żeby przetestować warianty
    pole_min_days_values = [4]             # cofnięte do 4 — krótkie maszty generują fałszywe sygnały
    pole_max_days_values = [40]            # Bulkowski: 2 miesiące sesyjne
    pole_min_growth_values = [0.85]        # Bulkowski: "podwaja się lub prawie" — próg 85%
    pole_max_daily_decline_values = [0.20]  # nieużywane, zachowane dla kompatybilności
    max_days_without_new_high_values = [3]  # po ilu dniach bez nowego High maszt się kończy
    flag_min_days_values = [3]             # Bulkowski: min 3 dni
    flag_max_days_until_breakout_values = [25]  # punkt 4: zluzowane z 19 do 25
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
        "actual_result_pct",       # idealny szczyt (spadek 20%) lub SL
        "trailing_10pct_result",   # trailing stop 20% od szczytu
        "trailing_atr_result",     # trailing stop 2*ATR od szczytu
        "bulkowski_stop_result",   # stop zmienności jak Bulkowski
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
        "actual_result_pct_avg",
        "trailing_10pct_result_avg",
        "bulkowski_stop_result_avg",
        "bulkowski_stop_result_median",
        "bulkowski_stop_result_positive_rate",
        "stop_loss_hit_rate",
    ]

    available_cols = [col for col in cols if col in summary_df.columns]

    print("TOP konfiguracje (sortowane wg actual_result_pct):")
    print(summary_df[available_cols].head(top_n).to_string(index=False))


def main() -> None:
    # ============================================================
    # KONFIGURACJA — zmień tylko tę jedną linię żeby przełączyć rynek
    MARKET = "NYSE"   # "NYSE" lub "GPW"
    # ============================================================

    if MARKET == "NYSE":
        from data import ALL_US as tickers
        start_date    = datetime(2024, 8, 29)
        end_date      = datetime(2026, 8, 29)
        signal_from   = pd.Timestamp(datetime(2025, 8, 29))
        signal_cutoff = pd.Timestamp(datetime(2026, 8, 29))
    elif MARKET == "GPW":
        from data import ALL as tickers
        start_date    = datetime(1991, 1, 1)
        end_date      = datetime.today()
        signal_from   = None
        signal_cutoff = None
    else:
        raise ValueError(f"Nieznany rynek: {MARKET}")

    market = MARKET
    market_suffix = MARKET_SUFFIXES[market]

    print(f"Rynek: {MARKET} | Okres: {start_date.date()} → {end_date.date()}")
    print("Ładowanie danych z cache / dociąganie braków...")
    history_map: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
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

            detection_df = df.loc[df.index <= signal_cutoff] if signal_cutoff is not None else df
            rows_all = backtest_flag_for_ticker(
                ticker=ticker,
                df=detection_df,
                params=params,
            )
            # Filtruj sygnały spoza okna signal_from
            rows = [r for r in rows_all if signal_from is None or pd.Timestamp(r["date"]) >= signal_from]

            ticker_elapsed = perf_counter() - ticker_start
            config_elapsed = perf_counter() - config_start

            if rows:
                for row in rows:
                    actual = row.get("actual_result_pct")
                    trailing = row.get("trailing_10pct_result")
                    sl = row.get("stop_loss_hit")
                    max20 = row.get("max_gain_20d_pct")

                    def fmt(v: Any) -> str:
                        return f"{v:.1f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "n/a"

                    sl_str = "🔴SL" if sl else "🟢ok"
                    result_str = fmt(actual)
                    emoji = "✅" if actual is not None and actual > 0 else "❌"

                    print(
                        f"  {ticker} | {pd.Timestamp(row['date']).strftime('%Y-%m-%d')} | "
                        f"close={row['close_event']:.2f} | {sl_str} | "
                        f"wynik={emoji}{result_str} | trail20={fmt(trailing)} | bulSL={fmt(row.get('bulkowski_stop_result'))} | max20d={fmt(max20)} | "
                        f"{row['signal']} | {ticker_idx}/{total_tickers}"
                    )

                    PRINT_OHLC = False  # zmień na True żeby widzieć notowania dzień po dniu
                    if PRINT_OHLC:
                        flag_low_val = row.get("flag_low")
                        flag_days_val = row.get("flag_days")
                        try:
                            event_idx_val = df.index.get_loc(pd.Timestamp(row["date"]))
                            if isinstance(event_idx_val, int) and flag_days_val is not None:
                                fs_idx = event_idx_val - int(flag_days_val)
                                print(f"    {'Data':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}  Faza")
                                end_idx = min(event_idx_val + 61, len(df))
                                for di in range(max(0, fs_idx), end_idx):
                                    d = df.iloc[di]
                                    date_s = df.index[di].strftime("%Y-%m-%d")
                                    if di < event_idx_val:
                                        phase = "FLAGA"
                                    elif di == event_idx_val:
                                        phase = "WYBICIE"
                                    else:
                                        sl_marker = " ◄SL" if flag_low_val is not None and float(d["Low"]) < flag_low_val else ""
                                        phase = f"PO{sl_marker}"
                                    print(f"    {date_s:<12} {float(d['Open']):>8.2f} {float(d['High']):>8.2f} {float(d['Low']):>8.2f} {float(d['Close']):>8.2f}  {phase}")
                                print()
                        except Exception:
                            pass

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
        ["actual_result_pct_avg", "max_gain_20d_pct_avg", "change_10d_pct_avg"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    details_df = pd.DataFrame(all_detail_rows)
    if not details_df.empty:
        details_df = details_df.sort_values(["config", "ticker", "date"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flag_param_search_summary_{market}_{timestamp}.csv"
    details_path = f"/Users/pl8000269/IdeaProjects/rss_gpw/back_tests/reports/flag_param_search_details_{market}_{timestamp}.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_top_configs(summary_df)

    print()
    print(f"Zapisano summary: {summary_path}")
    print(f"Zapisano details: {details_path}")


if __name__ == "__main__":
    main()