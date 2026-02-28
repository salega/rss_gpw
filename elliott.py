import sys
import yfinance as yf
from operator import itemgetter
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional

from datetime import  datetime, timedelta
import os

from send_email import send_email
from data import SWIG_80, MWIG_40, WIG_20

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_last_year_price_data(company_abbr: str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download(company_abbr + ".WA", start=start_date, end=end_date, progress=False)
    prices = data['Close'][company_abbr + ".WA"].to_dict()

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
        ema_above(21),
        ema_above(30),
    )


def calculate_potential(company_abbr: str):
    prices = get_last_year_price_data(company_abbr)
    price_above_emas = check_if_price_above_emas(prices)
    max = get_max_value(prices)
    last = next(reversed(prices.items()))
    penultimate = list(prices.items())[-2]
    max_value = float(max[1])
    last_value = float(last[1])
    penultimate_value = float(penultimate[1])
    max_date = max[0]

    seen_max = False
    local_min_value = max_value  
    for dt, val in prices.items():
        if not seen_max:
            if dt == max_date:
                seen_max = True
            continue

        if float(val) < local_min_value:
            local_min_value = float(val)

    max_40_percent_greater_than_local_min = max_value > local_min_value * 1.4
    today_between_10_and_50_percent_greater_than_local_min = last_value > local_min_value * 1.1 and last_value <= local_min_value * 1.5
    today_higher_than_yesterday = last_value > penultimate_value
    is_at_least_one_ema_above = any(price_above_emas)
    has_potential = (max_40_percent_greater_than_local_min and is_at_least_one_ema_above and
                     today_between_10_and_50_percent_greater_than_local_min and today_higher_than_yesterday)

    return {
            "company": company_abbr,
            "has_potential": has_potential,
            "above_ema_8": price_above_emas[0],
            "above_ema_21": price_above_emas[1],
            "above_ema_30": price_above_emas[2],
            "max_value": f"{max_value:.2f}", 
            "local_min_value": f"{local_min_value:.2f}",
            "last_value": f"{last_value:.2f}",
            "penultimate_value": f"{penultimate_value:.2f}"
            }


def format_potential(potential):
    stooq_url = f'https://stooq.pl/q/?s={potential["company"]}'
    ema_8_icon = "✅️" if potential["above_ema_8"] else "❌"
    ema_21_icon = "✅️" if potential["above_ema_21"] else "❌"
    ema_30_icon = "✅️" if potential["above_ema_30"] else "❌"

    formatted_entry = f"""\
<div style="font-size: 0.88em; margin-top: 20px; padding: 0; line-height: 1.5;">
  <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin: 0; padding: 0;">
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">🏭<b>{potential["company"]}</b></td>
      <td style="padding: 0 14px 0 0;"><span style="font-size: 0.9em;">[<a href="{stooq_url}">stooq</a>]</span></td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">📈EMA8: {ema_8_icon}</td>
      <td style="padding: 0 14px 0 0;">📈EMA21: {ema_21_icon}</td>
      <td style="padding: 0;">📈EMA30: {ema_30_icon}</td>
    </tr>
    <tr style="margin: 0; padding: 0;">
      <td style="padding: 0 14px 0 0;">⬆️{potential["max_value"]}</td>
      <td style="padding: 0 14px 0 0;">⬇️{potential["local_min_value"]}</td>
      <td style="padding: 0;">🆕{potential["penultimate_value"]} → {potential["last_value"]}</td>
    </tr>
  </table>
</div>"""
    return formatted_entry


if __name__ == "__main__":

    if os.environ["RUN_FOR_ALL_COMPANIES"] == "true":
        company_abbr = ""
    else:
        company_abbr = input("\nPodaj skrót spółki lub puste: ").strip().upper()

    if company_abbr == "":
        report = '<span style="font-size: 1.6em;"><b>🏪SWIG80:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in SWIG_80:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                potential = format_potential(potential)
                report = report + potential #+ "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        report = report + '<br><br><span style="font-size: 1.6em;"><b>🏬MWIG40:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in MWIG_40:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                potential = format_potential(potential)
                report = report + potential# + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        report = report + '<br><br><span style="font-size: 1.6em;"><b>🏢WIG20:</b></span><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        for company in WIG_20:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                potential = format_potential(potential)
                report = report + potential

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
