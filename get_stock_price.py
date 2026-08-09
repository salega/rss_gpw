"""
Pobiera i wypisuje notowania OHLCV dla podanej spółki i okresu.

Użycie:
    python3 get_stock_price.py
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def main() -> None:
    # --- DEBUG: zahardkodowane wartości dla NX ---
    ticker    = "NX"
    suffix    = ""          # US — bez sufiksu
    start_str = "2009-12-01"
    end_str   = "2010-04-01"
    # Kluczowe daty formacji:
    # L1=2010-02-10 @ 12.05, Neck=2010-02-22 @ 14.25, L2=2010-02-25 @ 11.93
    # Breakout=2010-03-01 @ 14.25, result=+21.9% (peak: 17.37)
    MARKERS = {
        "2010-02-10": "← L1      @ 12.05",
        "2010-02-22": "← NECK    @ 14.25",
        "2010-02-25": "← L2      @ 11.93",
        "2010-03-01": "← BREAKOUT @ 14.25",
    }
    # --- koniec DEBUG ---

    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d") + timedelta(days=1)

    symbol = ticker + suffix
    print(f"\nPobieram: {symbol}  {start_str} → {end_str} ...\n")

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        print("Brak danych.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=-1)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna().sort_index()

    print(f"{'Data':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 68)
    for date, row in df.iterrows():
        vol    = f"{int(row['Volume']):,}" if "Volume" in row and pd.notna(row["Volume"]) else "n/a"
        marker = MARKERS.get(str(date.date()), "")
        print(
            f"{str(date.date()):<12}"
            f" {float(row['Open']):>10.3f}"
            f" {float(row['High']):>10.3f}"
            f" {float(row['Low']):>10.3f}"
            f" {float(row['Close']):>10.3f}"
            f" {vol:>12}"
            f"  {marker}"
        )

    print(f"\nŁącznie: {len(df)} sesji")


if __name__ == "__main__":
    main()
