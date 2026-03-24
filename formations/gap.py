from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def get_if_theoretical_open_is_bearish_gap_vs_yesterday_close(
        company_abbr: str,
        min_gap_pct: float = 2.0
):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=10)  # mały bufor na weekendy/święta

        data = yf.download(company_abbr + ".WA", start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(company_abbr + ".WA", axis=1, level=-1)

        data = data[["Close"]].dropna()
        if data.empty:
            return None

        yesterday_close = float(data.iloc[-1]["Close"])

        theoretical_open = get_tko(company_abbr)

        if yesterday_close <= 0:
            return None

        gap_pct = ((theoretical_open / yesterday_close) - 1.0) * 100.0  # ujemne = TKO niżej niż close

        if gap_pct <= -abs(min_gap_pct):
            return gap_pct

        return None
    except Exception:
        return None


def get_tko(company_abbr):
    return 5.0


def get_last_30_days_open_close(company_abbr: str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=45)  # buffer na weekendy/święta, ale filtrujemy do 30 dni kalendarzowych

    data = yf.download(company_abbr + ".WA", start=start_date, end=end_date, progress=False)

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(company_abbr + ".WA", axis=1, level=-1)

    data = data[["Open", "Close"]].dropna()
    if data.empty:
        return None

    cutoff = (end_date - timedelta(days=30)).date()
    data = data[data.index.date >= cutoff]
    if data.empty:
        return None

    return data


def find_bearish_open_gaps_last_30_days(company_abbr: str, min_gap_pct: float = 2.0):
    """
    Szuka spadkowej luki otwarcia w ostatnich 30 dniach:
    Open <= Close(prev_day) * (1 - min_gap_pct/100)

    Dla każdego dnia z luką liczy "wejście" wg reguły:
    kupno w momencie, gdy maksimum świecy 1-min przekroczy maksimum poprzedniej świecy 1-min
    (czyli sygnał wybicia z poprzedniego minutowego słupka).

    Następnie liczy intraday_pct jako: (Close(dnia) / entry_price - 1) * 100.
    Jeżeli w danym dniu nie ma sygnału (brak wybicia), pomija ten dzień.
    """
    oc = get_last_30_days_open_close(company_abbr)
    if oc is None or len(oc) < 2:
        return []

    results = []

    prev_close = oc["Close"].shift(1)
    gap_pct_series = ((oc["Open"] / prev_close) - 1.0) * 100.0  # ujemne = luka w dół
    gap_mask = gap_pct_series <= -abs(min_gap_pct)

    for dt, has_gap in gap_mask.items():
        if pd.isna(has_gap) or not bool(has_gap):
            continue

        prev_close_price = float(prev_close.loc[dt])
        gap_pct = float(gap_pct_series.loc[dt])

        # Pobieramy 1-min dane dla dnia luki
        day_start = dt.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        intraday_1m = yf.download(
            company_abbr + ".WA",
            start=day_start,
            end=day_end,
            interval="1m",
            progress=False
        )
        if intraday_1m is None or intraday_1m.empty:
            continue

        if isinstance(intraday_1m.columns, pd.MultiIndex):
            intraday_1m = intraday_1m.xs(company_abbr + ".WA", axis=1, level=-1)

        intraday_1m = intraday_1m[["High"]].dropna()
        if len(intraday_1m) < 2:
            continue

        # Sygnał: High(t) > High(t-1) -> kupujemy po High(t-1) (pierwszy możliwy breakout)
        prev_high = intraday_1m["High"].shift(1)
        signal_mask = intraday_1m["High"] > prev_high

        first_signal_ts = None
        for ts, ok in signal_mask.items():
            if pd.isna(ok) or not bool(ok):
                continue
            first_signal_ts = ts
            break

        if first_signal_ts is None:
            # brak wybicia poprzedniego maksimum -> brak transakcji wg reguły
            continue

        entry_price = float(prev_high.loc[first_signal_ts])

        # Close dnia: bierzemy dzienne dane tylko dla tego dnia, żeby mieć oficjalne close
        day_daily = yf.download(company_abbr + ".WA", start=day_start, end=day_end, progress=False)
        if day_daily is None or day_daily.empty:
            continue

        if isinstance(day_daily.columns, pd.MultiIndex):
            day_daily = day_daily.xs(company_abbr + ".WA", axis=1, level=-1)

        day_daily = day_daily[["Open", "Close"]].dropna()
        if day_daily.empty:
            continue

        day_open = float(day_daily.iloc[0]["Open"])
        day_close = float(day_daily.iloc[0]["Close"])

        intraday_pct = ((day_close / entry_price) - 1.0) * 100.0

        results.append({
            "company": company_abbr,
            "date": dt.date().isoformat(),
            "prev_close": prev_close_price,
            "open": day_open,
            "close": day_close,
            "gap_pct": gap_pct,
            "entry_price": entry_price,
            "entry_time": first_signal_ts.isoformat(),
            "intraday_pct": intraday_pct,
        })

    return results