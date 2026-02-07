
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import warnings
import yfinance as yf
import pandas as pd

warnings.simplefilter("ignore", category=UserWarning)


swig80_dict = {
    "ELEKTROTI": "ELT",
    "AGORA": "AGO",
    "AMBRA": "AMB",
    "AMICA": "AMC",
    "APATOR": "APT",
    "ASTARTA": "AST",
    "BIOTON": "BIO",
    "BORYSZEW": "BRS",
    "BOS": "BOS",
    "COMP": "CMP",
    "DECORA": "DCR",
    "ECHO": "ECH",
    "ERBUD": "ERB",
    "GRENEVIA": "GEA",
    "FORTE": "FTE",
    "KOGENERA": "KGN",
    "MCI": "MCI",
    "MENNICA": "MNC",
    "MOSTALZAB": "MSZ",
    "POLIMEXMS": "PXM",
    "SANOK": "SNK",
    "SNIEZKA": "SKA",
    "STALEXP": "STX",
    "STALPROD": "STP",
    "SYGNITY": "SGN",
    "VRG": "VRG",
    "WAWEL": "WWL",
    "MCR": "MCR",
    "OPONEO": "OPN",
    "ASSECOBS": "ABS",
    "CIGAMES": "CIG",
    "WIELTON": "WLT",
    "UNIBEP": "UNI",
    "SELENAFM": "SEL",
    "QUERCUS": "QRS",
    "BUMECH": "BMC",
    "COGNOR": "COG",
    "BOGDANKA": "LWB",
    "ARCTIC": "ATC",
    "FERRO": "FRO",
    "MABION": "MAB",
    "VOTUM": "VOT",
    "COLUMBUS": "CLC",
    "BLOOBER": "BLO",
    "RYVU": "RVU",
    "SNTVERSE": "SNT",
    "ACAUTOGAZ": "ACG",
    "TOYA": "TOA",
    "AILLERON": "ALL",
    "MEDICALG": "MDG",
    "DATAWALK": "DAT",
    "UNIMOT": "UNT",
    "ZEPAK": "ZEP",
    "TARCZYNSKI": "TAR",
    "MLPGROUP": "MLG",
    "MERCATOR": "MRC",
    "PCCROKITA": "PCR",
    "TORPOL": "TOR",
    "VIGOPHOTN": "VGO",
    "ATAL": "1AT",
    "PEKABEX": "PBX",
    "WITTCHEN": "WTN",
    "ENTER": "ENT",
    "ARCHICOM": "ARH",
    "CLNPHARMA": "CLN",
    "PLAYWAY": "PLW",
    "SCPFL": "SCP",
    "XTPL": "XTP",
    "MLSYSTEM": "MLS",
    "CREEPYJAR": "CRJ",
    "SELVITA": "SLV",
    "DADELO": "DAD",
    "CAPTORTX": "CTX",
    "SHOPER": "SHO",
    "ONDE": "OND",
    "CREOTECH": "CRI",
    "BIOCELTIX": "BCX",
    "GREENX": "GRX",
    "MURAPOL": "MUR",
    "ARLEN": "ARL"
}


def normalize_gpw_ticker(user_symbol: str) -> str:
    sym = user_symbol.strip().upper()
    if sym.endswith(".WA"):
        return sym
    return f"{sym}.WA"

def fetch_last_n_sessions(ticker: str, n: int = 20):
    df = yf.download(
        tickers=ticker,
        period="90d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if df is None or df.empty:
        raise ValueError(f"Brak danych dla '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df[ticker]
        else:
            raise ValueError(f"Nie znaleziono kolumn dla tickera '{ticker}' w pobranych danych.")

    close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if close_col is None:
        raise ValueError(f"Brak kolumny Close/Adj Close dla '{ticker}'.")
    if "Volume" not in df.columns:
        raise ValueError(f"Brak kolumny Volume dla '{ticker}'.")

    df = df.dropna(subset=[close_col, "Volume"])
    if df.empty:
        raise ValueError(f"Brak kompletnych wierszy (Close/Volume) dla '{ticker}'.")

    df = df.rename(columns={close_col: "Close"})
    recent = df.tail(n)
    return recent

def average_turnover_last_n_days(company_abbr: str, n: int = 20):
    ticker = normalize_gpw_ticker(company_abbr)
    t = yf.Ticker(ticker)
    info = t.info if hasattr(t, "info") else {}
    currency = info.get("currency", "UNKNOWN")
    recent = fetch_last_n_sessions(ticker, n=n)
    turnover = (recent["Close"] * recent["Volume"]).astype(float)
    avg_turnover = float(turnover.mean())
    return avg_turnover, currency, ticker

def format_currency(value: float, currency: str) -> str:
    if value >= 1_000_000:
        return f"{value:,.0f} {currency}".replace(",", " ").replace(".", ",")
    return f"{value:,.2f} {currency}".replace(",", " ").replace(".", ",")

def main():
    if len(sys.argv) < 2:
        calculate_avg_turnover_for_all()
        sys.exit(0)
    try:
        company_abbr = sys.argv[1]
        calculate_avg_turnover_for(company_abbr, company_abbr)
    except Exception as e:
        print(f"Błąd: {e}")
        sys.exit(2)


def calculate_avg_turnover_for_all():
    for company, company_abbr in swig80_dict.items():
        calculate_avg_turnover_for(company, company_abbr)


def calculate_avg_turnover_for(company: str, company_abbr: str):
    try:
        avg, curr, yf_ticker = average_turnover_last_n_days(company_abbr, n=20)
        acceptable_avg_turnover = "[OK]" if avg > 1000000 else "[X]"
        print(f"{acceptable_avg_turnover} {company} [{yf_ticker}]: {format_currency(avg, curr)}")
    except Exception as e:
        print(f"Błąd: {e} dla {company} [{company_abbr}]")


if __name__ == "__main__":
    main()
