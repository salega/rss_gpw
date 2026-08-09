import os
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from data import SWIG_80, MWIG_40, WIG_20, SP_500, RUSSELL_2000
from formations.double_bottom import check_double_bottom_today
from formations.flag import check_flag_breakout_today
from formations.nr7 import check_nr7_confirmed_today
from send_email import send_email
from util import check_if_price_above_emas
from util import get_max_value

warnings.simplefilter(action='ignore', category=FutureWarning)
INCLUDE_DOWN_NR7 = False
DATE_TO_SIMULATE = None #datetime(2055, 10, 3)


def get_last_year_price_data(company_abbr: str, market_suffix: str = ".WA"):
    end_date = DATE_TO_SIMULATE if DATE_TO_SIMULATE else datetime.today()
    start_date = end_date - timedelta(days=365 * 2)  # 2 lata — double bottom potrzebuje więcej historii
    # yfinance end jest exclusive — dodajemy 1 dzień żeby end_date był włączony
    download_end = end_date + timedelta(days=1)

    ticker_symbol = company_abbr + market_suffix
    data = yf.download(ticker_symbol, start=start_date, end=download_end, progress=False, auto_adjust=True)

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker_symbol, axis=1, level=-1)

    cols = [c for c in ["Open", "Close", "High", "Low", "Volume"] if c in data.columns]
    prices = data[cols].dropna().to_dict(orient="index")

    return prices


def calculate_potential(company_abbr: str, market_suffix: str = ".WA"):
    prices = get_last_year_price_data(company_abbr, market_suffix=market_suffix)

    if prices is None:
        return None

    close_prices = {dt: float(v["Close"]) for dt, v in prices.items() if v and "Close" in v and v["Close"] is not None}

    if len(close_prices) < 2:
        return None

    price_above_emas = check_if_price_above_emas(close_prices)
    max = get_max_value(close_prices)
    last = next(reversed(close_prices.items()))
    penultimate = list(close_prices.items())[-2]
    max_value = float(max[1])
    last_value = float(last[1])
    penultimate_value = float(penultimate[1])
    max_date = max[0]

    seen_max = False
    local_min_value = max_value
    for dt, val in close_prices.items():
        if not seen_max:
            if dt == max_date:
                seen_max = True
            continue

        if float(val) < local_min_value:
            local_min_value = float(val)

    is_gpw = market_suffix == ".WA"

    nr7 = check_nr7_confirmed_today(prices)
    # rectangle_breakout_today = check_rectangle_breakout_today_daily_scan(prices) if is_gpw else None
    # flat_base_breakout_today = check_flat_base_breakout_today(prices) if is_gpw else None
    rectangle_breakout_today = None
    flat_base_breakout_today = None
    flag_breakout_today = check_flag_breakout_today(prices)
    double_bottom_breakout_today = check_double_bottom_today(
        prices,
        max_separation_days=100,
        max_bottom_diff_pct=0.06,
        min_peak_rise_pct=0.17,
        min_downtrend_pct=0.15,
        drop_into_l1_lookback=20,
        max_throwback_in_decline=0.08,
        check_volume=False,
    )

    return {
        "company": company_abbr,
        "market_suffix": market_suffix,
        "above_ema_8": price_above_emas[0],
        "above_ema_30": price_above_emas[1],
        "above_ema_200": price_above_emas[2],
        "max_value": f"{max_value:.2f}",
        "local_min_value": f"{local_min_value:.2f}",
        "last_value": f"{last_value:.2f}",
        "penultimate_value": f"{penultimate_value:.2f}",
        "nr7": nr7,
        "rectangle_breakout_today": rectangle_breakout_today,
        "flat_base_breakout_today": flat_base_breakout_today,
        "flag_breakout_today": flag_breakout_today,
        "double_bottom_breakout_today": double_bottom_breakout_today,
    }


def format_potential(potential):
    company = potential["company"]
    suffix = potential.get("market_suffix", ".WA")
    if suffix == ".WA":
        stooq_url = f'https://stooq.pl/q/?s={company}'
    else:
        stooq_url = f'https://finance.yahoo.com/quote/{company}'
    ema_8_icon = "✅️" if potential["above_ema_8"] else "❌"
    ema_30_icon = "✅️" if potential["above_ema_30"] else "❌"
    ema_200_icon = "✅️" if potential["above_ema_200"] else "❌"

    nr7 = potential.get("nr7") or ""

    def build_row(content):
        if not content:
            return ""
        rows = []
        for line in str(content).splitlines():
            if not line.strip():
                continue
            rows.append(
                f'<tr style="margin: 0; padding: 0;"><td colspan="3" style="padding: 0;">{line}</td></tr>'
            )

        return "".join(rows)

    rectangle_breakout = build_row(potential.get("rectangle_breakout_today"))
    flat_base_breakout = build_row(potential.get("flat_base_breakout_today"))
    flag_breakout = build_row(potential.get("flag_breakout_today"))
    double_bottom_breakout = build_row(potential.get("double_bottom_breakout_today"))

    formatted_entry = f"""\
<div style="font-size: 0.88em; margin-top: 20px; padding: 0; line-height: 1.5;">
  <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin: 0; padding: 0;">
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">🏭<b>{potential["company"]}</b></td>
      <td style="padding: 0 14px 0 0;"><span style="font-size: 0.9em;">[<a href="{stooq_url}">stooq</a>]</span></td>
      <td style="padding: 0;">{nr7}</td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">📈EMA8: {ema_8_icon}</td>
      <td style="padding: 0 14px 0 0;">📈EMA30: {ema_30_icon}</td>
      <td style="padding: 0;">📈EMA200: {ema_200_icon}</td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">⬆️{potential["max_value"]}</td>
      <td style="padding: 0 14px 0 0;">⬇️{potential["local_min_value"]}</td>
      <td style="padding: 0;">🆕{potential["penultimate_value"]} → {potential["last_value"]}</td>
    </tr>
    {rectangle_breakout}
    {flat_base_breakout}
    {flag_breakout}
    {double_bottom_breakout}
  </table>
</div>"""
    return formatted_entry


_progress_idx = 0
_progress_total = 0


def _log_progress(company: str, market_suffix: str, hit: bool) -> None:
    global _progress_idx
    _progress_idx += 1
    pct = int(_progress_idx / _progress_total * 100) if _progress_total else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    signal = "✅" if hit else "  "
    print(f"{signal} [{_progress_idx:4d}/{_progress_total}] {bar} {pct:3d}%  {company}{market_suffix}", flush=True)


def get_if_has_potential(company, market_suffix: str = ".WA"):
    potential = calculate_potential(company, market_suffix=market_suffix)

    hit = bool(potential and (potential["rectangle_breakout_today"]
                              or potential["flat_base_breakout_today"] or potential["flag_breakout_today"]
                              or potential["double_bottom_breakout_today"]))
    _log_progress(company, market_suffix, hit)

    if hit:
        return format_potential(potential)
    return ""


if __name__ == "__main__":

    if os.environ["RUN_FOR_ALL_COMPANIES"] == "true":
        company_abbr = ""
    else:
        company_abbr = input("\nPodaj skrót spółki lub puste: ").strip().upper()
        if company_abbr:
            market = input("GPW (tak/nie, domyślnie nie): ").strip().lower()
            market_suffix = ".WA" if market in ("tak", "t", "y", "yes") else ""
            result = calculate_potential(company_abbr, market_suffix=market_suffix)
            if result:
                print(format_potential(result))
                print()
                for k, v in result.items():
                    print(f"  {k}: {v}")
            else:
                print("Brak danych lub brak sygnału")
            import sys; sys.exit(0)

    if company_abbr == "":
        from data import ALL_US as _ALL_US
        _progress_total = len(SWIG_80) + len(MWIG_40) + len(WIG_20) + len(_ALL_US)

        # --- GPW ---
        report = '<span style="font-size: 1.6em;"><b>🇵🇱 GPW</b></span><br>'
        report += '<span style="font-size: 1.3em;"><b>🏪SWIG80:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in SWIG_80:
            report = report + get_if_has_potential(company, market_suffix=".WA")
        report = report + '<br><br><span style="font-size: 1.3em;"><b>🏬MWIG40:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in MWIG_40:
            report = report + get_if_has_potential(company, market_suffix=".WA")
        report = report + '<br><br><span style="font-size: 1.3em;"><b>🏢WIG20:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in WIG_20:
            report = report + get_if_has_potential(company, market_suffix=".WA")

        # --- US ---
        report += '<br><br><span style="font-size: 1.6em;"><b>🇺🇸 US</b></span><br>'
        report += '<span style="font-size: 1.3em;"><b>📊 S&P 500:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in SP_500:
            report = report + get_if_has_potential(company, market_suffix="")
        report += '<br><br><span style="font-size: 1.3em;"><b>📊 Russell 2000:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in RUSSELL_2000:
            report = report + get_if_has_potential(company, market_suffix="")

        if os.environ["SEND_EMAIL"] == "true":
            send_email("Raport notowań", report, body_html=f"""\
            <html>
              <body style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 
              'Courier New', monospace;">{report}</body>
            </html>
            """)

        print(report)
        sys.exit(0)

    if len(company_abbr) == 0:
        print("Podaj poprawny skrót spółki")
        sys.exit(1)

    print(calculate_potential(company_abbr))