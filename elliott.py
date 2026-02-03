import sys
from datetime import datetime, timedelta, date
import yfinance as yf
from operator import itemgetter
import pandas as pd
import warnings
from typing import Dict, Tuple

from datetime import date, datetime, timedelta, timezone
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket


def send_email(body: str, body_html = None):
    msg = MIMEMultipart("alternative") if body_html else MIMEMultipart()
    msg["From"] = "kielarzu@gmail.com"
    msg["To"] = "kielarzu@gmail.com"
    msg["Subject"] = "Raport notowań z " + date.today().strftime("%d.%m.%Y")
    msg.attach(MIMEText(body, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    host = "smtp.gmail.com"
    port = 465
    username = "kielarzu@gmail.com"
    smtp_password = os.environ["SMTP_PASSWORD"] 

    attempts = 10
    base_backoff = 1.0
    timeout_sec = 5  # ważne: zabezpiecza przed wiszącym połączeniem

    last_err = None
    for i in range(1, attempts + 1):
        try:
            # per-connection timeout
            with smtplib.SMTP_SSL(host, port, timeout=timeout_sec) as server:
                server.login(username, smtp_password)
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())
            return
        except (smtplib.SMTPException, socket.timeout, OSError) as e:
            last_err = e
            # backoff z jitterem minimalnym
            sleep_s = min(60, base_backoff * (2 ** (i - 1)))
            time.sleep(sleep_s)
            continue
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"Nie udało się wysłać maila po {attempts} próbach: {last_err}")


warnings.simplefilter(action='ignore', category=FutureWarning)
SWIG_80 = [
"1AT",
"ABS",
"ACG",
"AGO",
"ALL",
"AMB",
"AMC",
"APT",
"ARH",
"ARL",
"AST",
"ATC",
"BCX",
"BIO",
"BLO",
"BMC",
"BOS",
"BRS",
"CIG",
"CLC",
"CLN",
"CMP",
"COG",
"CRI",
"CRJ",
"CTX",
"DAD",
"DAT",
"DCR",
"ECH",
"ELT",
"ENT",
"ERB",
"FRO",
"FTE",
"GEA",
"GRX",
"KGN",
"LWB",
"MAB",
"MCI",
"MCR",
"MDG",
"MLG",
"MLS",
"MNC",
"MRC",
"MSZ",
"MUR",
"OND",
"OPN",
"PBX",
"PCR",
"PLW",
"PXM",
"QRS",
"RVU",
"SCP",
"SEL",
"SGN",
"SHO",
"SKA",
"SLV",
"SNK",
"STP",
"STX",
"SVE",
"TAR",
"TOA",
"TOR",
"UNI",
"UNT",
"VGO",
"VOT",
"VRG",
"WLT",
"WTN",
"WWL",
"XTP",
"ZEP",
]

MWIG_40 = [
"11B",
"ABE",
"ACP",
"APR",
"ASB",
"ASE",
"ATT",
"BFT",
"BHW",
"BNP",
"CAR",
"CBF",
"CPS",
"DIA",
"DOM",
"DVL",
"EAT",
"ENA",
"EUR",
"GPP",
"GPW",
"HUG",
"ING",
"JSW",
"LBW",
"MBR",
"MIL",
"MRB",
"NEU",
"NWG",
"PEP",
"RBW",
"SNT",
"TEN",
"TPE",
"TXT",
"VOX",
"VRC",
"WPL",
"XTB"
]

WIG_20 = [
"ALE",
"ALR",
"BDX",
"CCC",
"CDR",
"DNP",
"KGH",
"KRU",
"KTY",
"LPP",
"MBK",
"OPL",
"PCO",
"PEO",
"PGE",
"PKN",
"PKO",
"PZU",
"SPL",
"ZAB"
]


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


def check_if_ema21_is_rising(data: Dict[pd.Timestamp, float]) -> Tuple[pd.Series, bool]:
    if not data:
        raise ValueError("Pusty zbiór danych")

    s = pd.Series(data).sort_index()
    ema21 = s.ewm(span=21, adjust=False, min_periods=21).mean()

    if ema21.notna().sum() < 2:
        # Za mało punktów z wyliczoną EMA, żeby ocenić kierunek
        return ema21, False

    is_rising = bool(ema21.iloc[-1] > ema21.iloc[-2] and ema21.iloc[-2] > ema21.iloc[-3] and ema21.iloc[-3] > ema21.iloc[-4])
    return is_rising


def calculate_potential(company_abbr: str):
    prices = get_last_year_price_data(company_abbr)
    ema21_is_rising = check_if_ema21_is_rising(prices)
    max = get_max_value(prices)
    min = get_min_value(prices)
    last = next(reversed(prices.items()))
    penultimate = list(prices.items())[-2]
    max_value = float(max[1])
    min_value = float(min[1])
    last_value = float(last[1])
    penultimate_value = float(penultimate[1])
    max_date = max[0]
    min_date = min[0]

    is_70_percent_higher = max_value >= min_value * 1.6
    max_is_before_min = max_date < min_date
    last_20_percent_higher_than_min = last_value >= min_value * 1.1
    today_higher_than_yesterday = last_value > penultimate_value
    has_potential = is_70_percent_higher and max_is_before_min and last_20_percent_higher_than_min and ema21_is_rising

    return {
            "company": company_abbr,
            "has_potential": has_potential,
            "today_higher_than_yesterday": today_higher_than_yesterday
            }


if __name__ == "__main__":

    company_abbr = "ALL" #input("\nPodaj skrót spółki lub napisz 'ALL': ").strip().upper()
    
    if company_abbr == "ALL":
        full_report = "SWIG_80:\n\n"
        for company in SWIG_80:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                full_report = full_report + str(potential) + "\n\n"
        full_report = full_report + "\nMWIG_40:\n\n"
        for company in MWIG_40:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                full_report = full_report + str(potential) + "\n\n"
        full_report = full_report + "\nWIG_20:\n\n"
        for company in WIG_20:
            potential = calculate_potential(company)
            if potential["has_potential"]:
                full_report = full_report + str(potential) + "\n\n"

        #print(full_report)
        send_email(full_report)
        sys.exit(0)

    if len(company_abbr) == 0:
        print("Podaj poprawny skrót spółki")
        sys.exit(1)

    print(calculate_potential(company_abbr))
    
