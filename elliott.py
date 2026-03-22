import sys
import yfinance as yf
from operator import itemgetter
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional

from datetime import datetime, timedelta
import os

from send_email import send_email
from formations.flag import check_flag_breakout_today
from formations.rectangle import check_rectangle_breakout_today
from formations.flat_base import check_flat_base_breakout_today
from formations.nr7 import check_nr7_confirmed_today
from data import SWIG_80, MWIG_40, WIG_20

warnings.simplefilter(action='ignore', category=FutureWarning)
INCLUDE_DOWN_NR7 = False


def get_last_year_price_data(company_abbr: str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download(company_abbr + ".WA", start=start_date, end=end_date, progress=False)

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(company_abbr + ".WA", axis=1, level=-1)

    prices = data[["Close", "High", "Low"]].dropna().to_dict(orient="index")

    return prices


def get_max_value(prices):
    if not prices:
        return None
    return max(prices.items(), key=itemgetter(1))


def get_min_value(prices):
    if not prices:
        return None
    return min(prices.items(), key=itemgetter(1))


def check_if_price_above_emas(data: Dict[pd.Timestamp, float]) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if not data:
        raise ValueError("Pusty zbiór danych")

    s = pd.Series(data).sort_index()
    last_price = s.iloc[-1]

    def ema_above(span: int) -> Optional[bool]:
        ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
        if pd.isna(ema.iloc[-1]):
            return None
        return bool(last_price > ema.iloc[-1])

    return (
        ema_above(8),
        ema_above(30),
        ema_above(200),
    )


def calculate_potential(company_abbr: str):
    prices = get_last_year_price_data(company_abbr)
    close_prices = {dt: float(v["Close"]) for dt, v in prices.items() if v and "Close" in v and v["Close"] is not None}

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

    max_40_percent_greater_than_local_min = max_value > local_min_value * 1.4
    today_between_10_and_50_percent_greater_than_local_min = local_min_value * 1.1 < last_value <= local_min_value * 1.5
    today_higher_than_yesterday = last_value > penultimate_value
    is_at_least_one_ema_above = price_above_emas[0] or price_above_emas[1]
    has_potential = (max_40_percent_greater_than_local_min and is_at_least_one_ema_above and
                     today_between_10_and_50_percent_greater_than_local_min and today_higher_than_yesterday)
    nr7 = check_nr7_confirmed_today(prices)
    rectangle_breakout_today = check_rectangle_breakout_today(prices)
    flat_base_breakout_today = check_flat_base_breakout_today(prices)
    flag_breakout_today = check_flag_breakout_today(company_abbr, prices)

    return {
        "company": company_abbr,
        "has_potential": has_potential,
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
        "flag_breakout_today": flag_breakout_today
    }


def format_potential(potential):
    stooq_url = f'https://stooq.pl/q/?s={potential["company"]}'
    ema_8_icon = "✅️" if potential["above_ema_8"] else "❌"
    ema_30_icon = "✅️" if potential["above_ema_30"] else "❌"
    ema_200_icon = "✅️" if potential["above_ema_200"] else "❌"

    nr7 = ""
    if potential.get("nr7") == "UP":
        nr7 = "📊NR7: ⬆️"
    elif potential.get("nr7") == "DOWN":
        nr7 = "📊NR7: ⬇️"

    rectangle_breakout = ""
    if potential.get("rectangle_breakout_today"):
        rectangle_breakout = f"{potential['rectangle_breakout_today']}"

    flat_base_breakout = ""
    if potential.get("flat_base_breakout_today"):
        flat_base_breakout = f"{potential['flat_base_breakout_today']}"

    flag_breakout = ""
    if potential.get("flag_breakout_today"):
        flag_breakout = f"{potential['flag_breakout_today']}"

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
    <tr style="margin: 0; padding: 0;">
      <td colspan="3" style="padding: 6px 0 0 0;">{rectangle_breakout}</td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td colspan="3" style="padding: 6px 0 0 0;">{flat_base_breakout}</td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td colspan="3" style="padding: 6px 0 0 0;">{flag_breakout}</td>
    </tr>
  </table>
</div>"""
    return formatted_entry


def get_if_has_potential(company):
    potential = calculate_potential(company)
    if (potential["has_potential"] or potential["nr7"] or potential["rectangle_breakout_today"]
            or potential["flat_base_breakout_today"] or potential["flag_breakout_today"]):
        potential = format_potential(potential)
        return potential
    return ""


if __name__ == "__main__":

    if os.environ["RUN_FOR_ALL_COMPANIES"] == "true":
        company_abbr = ""
    else:
        company_abbr = input("\nPodaj skrót spółki lub puste: ").strip().upper()

    if company_abbr == "":
        report = '<span style="font-size: 1.6em;"><b>🏪SWIG80:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in SWIG_80:
            report = report + get_if_has_potential(company)
        report = report + '<br><br><span style="font-size: 1.6em;"><b>🏬MWIG40:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in MWIG_40:
            report = report + get_if_has_potential(company)
        report = report + '<br><br><span style="font-size: 1.6em;"><b>🏢WIG20:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in WIG_20:
            report = report + get_if_has_potential(company)

        report = report + """
        <br><br>
        <div style="font-size: 0.95em; margin-top: 10px; line-height: 1.4;">
          <b>Pamiętaj, że fala 3 powinna:</b>
          <ul style="margin: 6px 0 0 18px; padding: 0;">
            <li>zaczynać się po 38–70% fali 2 (która jest falą ABC)</li>
            <li>mieć większy wolumen</li>
            <li>mieć bardziej dynamiczny start</li>
          </ul>
        </div>
        """

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
