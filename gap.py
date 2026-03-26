from typing import Optional
from datetime import datetime, timedelta
from data import SWIG_80, MWIG_40, WIG_20

import pandas as pd
import yfinance as yf
from pathlib import Path


def load_tko() -> dict[str, float]:
    today_str = datetime.today().strftime("%Y-%m-%d")
    xls_path = Path(__file__).resolve().parent / f"akcje_{today_str}.xls"
    if not xls_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {xls_path}")

    raw = xls_path.read_bytes()
    head = raw[:256].lstrip().lower()

    # HTML udający XLS (jak w Twoim przykładzie)
    if not (head.startswith(b"<html") or b"<table" in head):
        raise ValueError(f"Plik nie wygląda na HTML (a oczekiwany jest HTML): {xls_path}")

    import io
    text = raw.decode("utf-8", errors="ignore")
    tables = pd.read_html(io.StringIO(text))
    if not tables:
        raise ValueError(f"Nie znaleziono żadnej tabeli HTML w pliku: {xls_path}")

    # wybierz tabelę, która ma (po spłaszczeniu) kolumny Skrót i TKO
    df = None
    for t in tables:
        candidate = t.copy()
        if isinstance(candidate.columns, pd.MultiIndex):
            candidate.columns = [
                (str(c[1]).strip() if len(c) > 1 and str(c[1]).strip() and not str(c[1]).startswith("Unnamed") else str(c[0]).strip())
                for c in candidate.columns
            ]
        else:
            candidate.columns = [str(c).strip() for c in candidate.columns]

        candidate = candidate.loc[:, [c for c in candidate.columns if c and not c.startswith("Unnamed")]]

        if {"Skrót", "TKO"}.issubset(set(candidate.columns)):
            df = candidate
            break

    if df is None:
        raise ValueError(f"Nie znaleziono tabeli z kolumnami 'Skrót' i 'TKO' w pliku: {xls_path}")

    df = df.astype(str)

    required = {"Skrót", "TKO"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w {xls_path.name}: {sorted(missing)}")

    def parse_tko(v: str) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s == "-":
            return None
        s = s.replace("\xa0", "").replace(" ", "")

        if "," in s:
            s = s.replace(".", "")
            s = s.replace(",", ".")

        try:
            x = float(s)
            return x if x > 0 else None
        except ValueError:
            return None

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        abbr = (row.get("Skrót") or "").strip()
        tko = parse_tko(row.get("TKO"))
        if not abbr or tko is None:
            continue
        result[abbr] = tko

    return result


def get_if_theoretical_open_is_bearish_gap_vs_yesterday_close(
        company_abbr: str,
        tko_map: dict[str, float],
        min_gap_pct: float = 1.0
):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=10)  # bufor na weekendy/święta

        data = yf.download(
            company_abbr + ".WA",
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(company_abbr + ".WA", axis=1, level=-1)

        data = data[["Close"]].dropna()
        if data.empty:
            return None

        yesterday_close = float(data.iloc[-1]["Close"])
        if yesterday_close <= 0:
            return None

        theoretical_open = tko_map.get(company_abbr)
        if theoretical_open is None or theoretical_open <= 0:
            return None

        # Jeśli różnica > 100% (czyli theoretical_open jest <0.5x lub >2x yesterday_close),
        # to najpewniej skala jest x10000 -> koryguj.
        ratio = theoretical_open / yesterday_close
        if ratio > 2.0 or ratio < 0.5:
            theoretical_open = theoretical_open / 10000.0

        gap_pct = ((theoretical_open / yesterday_close) - 1.0) * 100.0  # ujemne = luka w dół


        print(f"{company_abbr}: {theoretical_open:.2f} - {yesterday_close:.2f} = {gap_pct:.2f}%")

        if gap_pct <= -abs(min_gap_pct):
            return gap_pct

        return None
    except Exception as ex:
        print(f"Błąd przy pobieraniu danych dla {company_abbr}: {str(ex)}")
        return None


if __name__ == "__main__":
    tko_map = load_tko()

    all_found: list[tuple[str, str, float]] = []  # (idx_name, ticker, gap_pct)

    for idx_name, companies in [("SWIG80", SWIG_80), ("MWIG40", MWIG_40), ("WIG20", WIG_20)]:
        print(f"\n{idx_name}:")
        any_found = False
        for company in companies:
            gap_pct = get_if_theoretical_open_is_bearish_gap_vs_yesterday_close(company, tko_map, min_gap_pct=2.0)
            if gap_pct is None:
                continue
            any_found = True
            all_found.append((idx_name, company, gap_pct))
            print(f"[LUKA ZNALEZIONA] {company}: {gap_pct:.2f}%")
        if not any_found:
            print("  Brak spółek spełniających warunek.")

    print("\nPODSUMOWANIE (spółki z luką):")
    if not all_found:
        print("  Brak spółek spełniających warunek.")
    else:
        for _, company, gap_pct in sorted(all_found, key=lambda x: x[2]):
            print(f"  {company} {gap_pct:.2f}%")
