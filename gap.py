from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from data import SWIG_80, MWIG_40, WIG_20

DEFAULT_LOOKBACK_DAYS = 10
SCALE_FIX_DIVISOR = 10000.0
SCALE_FIX_RATIO_MIN = 0.5
SCALE_FIX_RATIO_MAX = 2.0


@dataclass(frozen=True)
class GapResult:
    idx_name: str
    ticker: str
    gap_pct: float  # negative = bearish gap


def _is_html_disguised_xls(raw: bytes) -> bool:
    head = raw[:256].lstrip().lower()
    return head.startswith(b"<html") or b"<table" in head


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            (
                str(c[1]).strip()
                if len(c) > 1 and str(c[1]).strip() and not str(c[1]).startswith("Unnamed")
                else str(c[0]).strip()
            )
            for c in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]

    out = out.loc[:, [c for c in out.columns if c and not str(c).startswith("Unnamed")]]
    return out


def _parse_tko(value: object) -> Optional[float]:
    if value is None:
        return None

    s = str(value).strip()
    if not s or s == "-":
        return None

    s = s.replace("\xa0", "").replace(" ", "")

    # 1 234,56 / 1.234,56 / 1234.56
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")

    try:
        x = float(s)
    except ValueError:
        return None

    return x if x > 0 else None


def load_tko(as_of: Optional[datetime] = None) -> dict[str, float]:
    as_of = as_of or datetime.today()
    today_str = as_of.strftime("%Y-%m-%d")
    xls_path = Path(__file__).resolve().parent / f"akcje_{today_str}.xls"

    if not xls_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {xls_path}")

    raw = xls_path.read_bytes()
    if not _is_html_disguised_xls(raw):
        raise ValueError(f"Plik nie wygląda na HTML (a oczekiwany jest HTML): {xls_path}")

    text = raw.decode("utf-8", errors="ignore")
    tables = pd.read_html(io.StringIO(text))
    if not tables:
        raise ValueError(f"Nie znaleziono żadnej tabeli HTML w pliku: {xls_path}")

    df: Optional[pd.DataFrame] = None
    for t in tables:
        candidate = _flatten_columns(t)
        if {"Skrót", "TKO"}.issubset(set(candidate.columns)):
            df = candidate
            break

    if df is None:
        raise ValueError(f"Nie znaleziono tabeli z kolumnami 'Skrót' i 'TKO' w pliku: {xls_path}")

    df = df.astype(str)

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        abbr = (row.get("Skrót") or "").strip()
        tko = _parse_tko(row.get("TKO"))
        if abbr and tko is not None:
            result[abbr] = tko

    return result


def _download_yesterday_close(company_abbr: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Optional[float]:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    data = yf.download(
        company_abbr + ".WA",
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        )
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(company_abbr + ".WA", axis=1, level=-1)

    data = data[["Close"]].dropna()
    if data.empty:
        return None

    close = float(data.iloc[-1]["Close"])
    return close if close > 0 else None


def _normalize_theoretical_open(theoretical_open: float, yesterday_close: float) -> float:
    ratio = theoretical_open / yesterday_close
    if ratio > SCALE_FIX_RATIO_MAX or ratio < SCALE_FIX_RATIO_MIN:
        return theoretical_open / SCALE_FIX_DIVISOR
    return theoretical_open


def _compute_gap_pct(theoretical_open: float, yesterday_close: float) -> float:
    return ((theoretical_open / yesterday_close) - 1.0) * 100.0


def get_if_theoretical_open_is_bearish_gap_vs_yesterday_close(
        company_abbr: str,
        tko_map: dict[str, float],
        min_gap_pct: float = 3.0,
) -> Optional[float]:
    yesterday_close = _download_yesterday_close(company_abbr)
    if yesterday_close is None:
        return None

    theoretical_open = tko_map.get(company_abbr)
    if theoretical_open is None or theoretical_open <= 0:
        return None

    theoretical_open = _normalize_theoretical_open(theoretical_open, yesterday_close)
    gap_pct = _compute_gap_pct(theoretical_open, yesterday_close)  # negative = gap down

    print(f"{company_abbr}: tko={theoretical_open:.2f} close={yesterday_close:.2f} gap={gap_pct:.2f}%")

    if gap_pct <= -abs(min_gap_pct):
        return gap_pct
    return None


def scan_indices_for_bearish_gaps(
        tko_map: dict[str, float],
        min_gap_pct: float = 2.0,
) -> list[GapResult]:
    results: list[GapResult] = []
    for idx_name, companies in [("SWIG80", SWIG_80), ("MWIG40", MWIG_40), ("WIG20", WIG_20)]:
        print(f"{idx_name}:")
        any_found = False
        for company in companies:
            try:
                gap_pct = get_if_theoretical_open_is_bearish_gap_vs_yesterday_close(company, tko_map, min_gap_pct)
            except Exception as ex:
                print(f"Błąd przy analizie {company}: {ex}")
                continue

            if gap_pct is None:
                continue

            any_found = True
            results.append(GapResult(idx_name=idx_name, ticker=company, gap_pct=gap_pct))
            print(f"[LUKA ZNALEZIONA] {company}: {gap_pct:.2f}%")

        if not any_found:
            print("  Brak spółek spełniających warunek.")

    return results


def main() -> int:
    tko_map = load_tko()
    results = scan_indices_for_bearish_gaps(tko_map, min_gap_pct=2.0)

    print("\nPODSUMOWANIE (spółki z luką):")
    if not results:
        print("  Brak spółek spełniających warunek.")
        return 0

    for r in sorted(results, key=lambda x: x.gap_pct):
        print(f"  {r.ticker} {r.gap_pct:+.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())