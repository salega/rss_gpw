from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Literal

import pandas as pd
import yfinance as yf

from data import SWIG_80, MWIG_40, WIG_20


DEFAULT_LOOKBACK_DAYS = 280


def _download_daily_history(company_abbr: str, lookback_days: int = 180) -> pd.DataFrame:
    df = yf.download(
        company_abbr + ".WA",
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(company_abbr + ".WA", axis=1, level=-1)

    required = {"Open", "Close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    return df[["Open", "Close"]].dropna()


def _download_intraday(company_abbr: str, start: datetime, end: datetime, interval: Literal["1m", "5m", "15m"]) -> pd.DataFrame:
    try:
        df = yf.download(
            company_abbr + ".WA",
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(company_abbr + ".WA", axis=1, level=-1)

    required = {"Close", "High"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    return df[["Close", "High"]].dropna()


def _is_bearish_open_gap(today_open: float, yesterday_close: float, min_gap_pct: float) -> Optional[float]:
    if today_open <= 0 or yesterday_close <= 0:
        return None
    gap_pct = ((today_open / yesterday_close) - 1.0) * 100.0
    return gap_pct if gap_pct <= -abs(min_gap_pct) else None


EntryRule = Literal[ "close_gt_prev_close"]


def _find_reversal_entry_from_intraday(df: pd.DataFrame, rule: EntryRule) -> Optional[tuple[float, pd.Timestamp]]:
    """
    Szukamy momentu wejścia:
      - 2 kolejne świece mają warunek spełniony (względem poprzedniej świecy),
      - kupujemy na Close drugiej świecy spełniającej warunek.
    Zwraca: (entry_price, entry_timestamp)
    """
    if df is None or df.empty:
        return None
    if not {"Close", "High"}.issubset(set(df.columns)):
        return None
    if len(df) < 3:
        return None

    consecutive = 0
    for i in range(1, len(df)):
        c = float(df["Close"].iloc[i])

        if rule == "close_gt_prev_high":
            ref = float(df["High"].iloc[i - 1])
        else:  # "close_gt_prev_close"
            ref = float(df["Close"].iloc[i - 1])

        if c > ref:
            consecutive += 1
            if consecutive >= 2:
                ts = df.index[i]
                return float(c), ts
        else:
            consecutive = 0

    return None


def _find_reversal_exit_from_intraday(df: pd.DataFrame) -> Optional[tuple[float, pd.Timestamp]]:
    """
    Wyjście gdy trend wzrostowy zaczyna się odwracać:
      - 2 kolejne świece mają Close < Close poprzedniej świecy,
      - sprzedajemy na Close drugiej spadkowej świecy.
    Zwraca: (exit_price, exit_timestamp)
    """
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    if len(df) < 3:
        return None

    consecutive_down = 0
    for i in range(1, len(df)):
        c = float(df["Close"].iloc[i])
        prev_c = float(df["Close"].iloc[i - 1])

        if c < prev_c:
            consecutive_down += 1
            if consecutive_down >= 2:
                ts = df.index[i]
                return float(c), ts
        else:
            consecutive_down = 0

    return None


def backtest_bearish_gap_reversal_strategy(
        companies: list[str],
        min_gap_pct: float,
        lookback_days: int,
        interval: Literal["1m", "5m", "15m"],
        entry_rule: EntryRule,
        intraday_max_days: int = 60,
        verbose_trades: bool = False,
) -> dict[str, float]:
    trade_returns_eod_pct: list[float] = []
    trade_returns_trendexit_pct: list[float] = []

    trades_count_eod = 0
    trades_count_trendexit = 0

    missing_daily_symbols = 0
    missing_intraday_days = 0
    gap_days = 0
    entry_signals = 0
    skipped_intraday_too_old = 0
    trend_exit_signals = 0
    missing_trend_exit = 0

    intraday_cutoff = datetime.today() - timedelta(days=intraday_max_days)

    for company in companies:
        daily = _download_daily_history(company, lookback_days=max(lookback_days + 80, 180))
        if daily.empty:
            missing_daily_symbols += 1
            continue

        daily = daily.tail(lookback_days + 1)  # +1 żeby mieć wczorajszy close
        if len(daily) < 2:
            continue

        dates = list(daily.index)
        for i in range(1, len(dates)):
            day = dates[i]
            prev_day = dates[i - 1]

            day_dt = day.to_pydatetime() if isinstance(day, pd.Timestamp) else day

            if day_dt < intraday_cutoff:
                skipped_intraday_too_old += 1
                continue

            try:
                today_open = float(daily.loc[day, "Open"])
                today_close = float(daily.loc[day, "Close"])
                yesterday_close = float(daily.loc[prev_day, "Close"])
            except Exception:
                continue

            gap_pct = _is_bearish_open_gap(today_open, yesterday_close, min_gap_pct=min_gap_pct)
            if gap_pct is None:
                continue

            gap_days += 1

            start = datetime(day_dt.year, day_dt.month, day_dt.day)
            end = start + timedelta(days=1)

            intraday = _download_intraday(company, start=start, end=end, interval=interval)
            if intraday.empty:
                missing_intraday_days += 1
                continue

            entry_found = _find_reversal_entry_from_intraday(intraday, rule=entry_rule)
            if entry_found is None:
                continue

            entry, entry_ts = entry_found
            if entry <= 0 or today_close <= 0:
                continue

            entry_signals += 1

            # 1) Exit EOD (na daily Close)
            ret_eod_pct = ((today_close / entry) - 1.0) * 100.0
            trade_returns_eod_pct.append(ret_eod_pct)
            trades_count_eod += 1

            # 2) Exit "trend reversal" (2 spadkowe świece intraday po wejściu)
            intraday_after_entry = intraday.loc[intraday.index >= entry_ts]
            exit_found = _find_reversal_exit_from_intraday(intraday_after_entry)
            if exit_found is None:
                missing_trend_exit += 1
            else:
                exit_price, exit_ts = exit_found
                if exit_price > 0:
                    trend_exit_signals += 1
                    ret_trendexit_pct = ((exit_price / entry) - 1.0) * 100.0
                    trade_returns_trendexit_pct.append(ret_trendexit_pct)
                    trades_count_trendexit += 1

            if verbose_trades:
                exit_found_dbg = _find_reversal_exit_from_intraday(intraday_after_entry)
                if exit_found_dbg is None:
                    exit_dbg = "trend_exit=NONE"
                else:
                    exit_p, exit_t = exit_found_dbg
                    exit_dbg = f"trend_exit@{exit_p:.2f} at {exit_t:%Y-%m-%d %H:%M}"

                print(
                    f"[TRADE] {company}.WA {day_dt:%Y-%m-%d} "
                    f"gap_open={today_open:.2f} (vs prev_close={yesterday_close:.2f}, gap={gap_pct:.2f}%) | "
                    f"rule={entry_rule} interval={interval} | "
                    f"buy@{entry:.2f} at {entry_ts:%Y-%m-%d %H:%M} | "
                    f"sell_eod@{today_close:.2f} | ret_eod={ret_eod_pct:.2f}% | {exit_dbg}"
                )

    def _summary(returns_pct: list[float]) -> dict[str, float]:
        if not returns_pct:
            return {"avg": 0.0, "median": 0.0, "win_rate": 0.0}
        s = pd.Series(returns_pct, dtype="float64")
        return {
            "avg": float(s.mean()),
            "median": float(s.median()),
            "win_rate": float((s > 0).mean() * 100.0),
        }

    eod = _summary(trade_returns_eod_pct)
    trend = _summary(trade_returns_trendexit_pct)

    return {
        "trades_eod": float(trades_count_eod),
        "avg_return_eod_pct": float(eod["avg"]),
        "median_return_eod_pct": float(eod["median"]),
        "win_rate_eod_pct": float(eod["win_rate"]),
        "trades_trendexit": float(trades_count_trendexit),
        "avg_return_trendexit_pct": float(trend["avg"]),
        "median_return_trendexit_pct": float(trend["median"]),
        "win_rate_trendexit_pct": float(trend["win_rate"]),
        "gap_days": float(gap_days),
        "entry_signals": float(entry_signals),
        "trend_exit_signals": float(trend_exit_signals),
        "missing_trend_exit": float(missing_trend_exit),
        "missing_daily_symbols": float(missing_daily_symbols),
        "missing_intraday_days": float(missing_intraday_days),
        "skipped_intraday_too_old": float(skipped_intraday_too_old),
    }


def main() -> int:
    companies = list(dict.fromkeys(SWIG_80 + MWIG_40 + WIG_20))

    min_gaps = [4.0, 5.0, 6.0]
    entry_rules: list[EntryRule] = ["close_gt_prev_close"]
    intervals: list[Literal["5m", "15m"]] = ["5m", "15m"]

    rows: list[dict[str, object]] = []

    total = len(min_gaps) * len(entry_rules) * len(intervals)
    idx = 0

    for min_gap in min_gaps:
        for rule in entry_rules:
            for interval in intervals:
                idx += 1
                print(
                    f"\n=== CONFIG {idx}/{total} === "
                    f"min_gap_pct={min_gap:.1f} | rule={rule} | interval={interval} | lookback_days={DEFAULT_LOOKBACK_DAYS} ==="
                )

                stats = backtest_bearish_gap_reversal_strategy(
                    companies=companies,
                    min_gap_pct=min_gap,
                    lookback_days=DEFAULT_LOOKBACK_DAYS,
                    interval=interval,
                    entry_rule=rule,
                    verbose_trades=False,
                )

                rows.append(
                    {
                        "min_gap_pct": min_gap,
                        "rule": rule,
                        "interval": interval,
                        "trades_eod": int(stats["trades_eod"]),
                        "avg_return_eod_pct": float(stats["avg_return_eod_pct"]),
                        "median_return_eod_pct": float(stats["median_return_eod_pct"]),
                        "win_rate_eod_pct": float(stats["win_rate_eod_pct"]),
                        "trades_trendexit": int(stats["trades_trendexit"]),
                        "avg_return_trendexit_pct": float(stats["avg_return_trendexit_pct"]),
                        "median_return_trendexit_pct": float(stats["median_return_trendexit_pct"]),
                        "win_rate_trendexit_pct": float(stats["win_rate_trendexit_pct"]),
                        "gap_days": int(stats["gap_days"]),
                        "entry_signals": int(stats["entry_signals"]),
                        "trend_exit_signals": int(stats["trend_exit_signals"]),
                        "missing_trend_exit": int(stats["missing_trend_exit"]),
                        "missing_daily_symbols": int(stats["missing_daily_symbols"]),
                        "missing_intraday_days": int(stats["missing_intraday_days"]),
                        "skipped_intraday_too_old": int(stats["skipped_intraday_too_old"]),
                    }
                )

    df = pd.DataFrame(rows)

    df = df.sort_values(["min_gap_pct", "interval", "rule"], ascending=[True, True, True])

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 260)

    print("\nPODSUMOWANIE (konfiguracje) - EXIT EOD vs EXIT trend-reversal:")
    print(df.to_string(index=False, justify="left"))

    df_rank = df[df["trades_eod"] >= 5].sort_values(["avg_return_eod_pct", "win_rate_eod_pct"], ascending=[False, False])
    if not df_rank.empty:
        print("\nTOP (trades_eod>=5) wg avg_return_eod_pct:")
        print(df_rank.head(10).to_string(index=False, justify="left"))

    df_rank2 = df[df["trades_trendexit"] >= 5].sort_values(
        ["avg_return_trendexit_pct", "win_rate_trendexit_pct"], ascending=[False, False]
    )
    if not df_rank2.empty:
        print("\nTOP (trades_trendexit>=5) wg avg_return_trendexit_pct:")
        print(df_rank2.head(10).to_string(index=False, justify="left"))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())