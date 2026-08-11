"""
Pobiera i wypisuje notowania OHLCV dla podanej spółki i okresu,
a następnie wyświetla interaktywny wykres świecowy (Plotly).

Użycie:
    python3 get_stock_price.py

    Po uruchomieniu wklej wiersz CSV z raportu (np. z double_bottom_param_search_details*.csv)
    i naciśnij Enter. Skrypt sam wyciągnie ticker, daty formacji i wyświetli wykres
    z marginesem 4 miesięcy po obu stronach formacji.

Kolumny CSV (0-based):
    9  = ticker
    12 = signal (etykieta formacji)
    14 = left_trough_date
    15 = right_trough_date
    16 = peak_date
    17 = left_trough_price
    18 = right_trough_price
    19 = peak_price
    20 = confirmation_price
    24 = stop_price
"""

import sys
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Margines po obu stronach formacji (dni kalendarzowych)
MARGIN_DAYS = 120


def parse_csv_row(line: str) -> dict:
    """
    Parsuje wiersz CSV z raportu — obsługuje double_bottom i scallop.
    Wykrywa format po sygnale (🔻 = double bottom, 🐚 = scallop).
    """
    # Użyj pandas do parsowania żeby obsłużyć cudzysłowy w sygnale
    from io import StringIO
    import csv
    reader = csv.reader(StringIO(line.strip()))
    parts = next(reader)

    if len(parts) < 20:
        raise ValueError(f"Za mało kolumn: {len(parts)}")

    # Wykryj format po sygnale
    # Double bottom: ticker na poz. 9, sygnał na 12
    # Scallop:       ticker na poz. 11, sygnał na 14
    signal_db = parts[12].strip() if len(parts) > 12 else ""
    signal_sc = parts[14].strip() if len(parts) > 14 else ""

    is_double_bottom = "🔻" in signal_db
    is_scallop = "🐚" in signal_sc or "🐚" in signal_db

    if is_double_bottom:
        return {
            "format":              "double_bottom",
            "ticker":              parts[9].strip(),
            "breakout_date":       parts[10].strip(),
            "breakout_price":      float(parts[11]),
            "signal":              parts[12].strip(),
            "left_trough_date":    parts[14].strip(),
            "right_trough_date":   parts[15].strip(),
            "peak_date":           parts[16].strip(),
            "left_trough_price":   float(parts[17]),
            "right_trough_price":  float(parts[18]),
            "peak_price":          float(parts[19]),
            "confirmation_price":  float(parts[20]),
            "stop_price":          float(parts[24]) if len(parts) > 24 else None,
        }
    elif is_scallop:
        return {
            "format":          "scallop",
            "ticker":          parts[11].strip(),
            "breakout_date":   parts[12].strip(),
            "breakout_price":  float(parts[13]),
            "signal":          parts[14].strip(),
            "a_date":          parts[15].strip(),
            "c_date":          parts[16].strip(),
            "b_date":          parts[17].strip(),
            "a_price":         float(parts[18]),
            "c_price":         float(parts[19]),
            "b_price":         float(parts[20]),
            "retracement_pct": float(parts[21]) if len(parts) > 21 else None,
            "ac_rise_pct":     float(parts[22]) if len(parts) > 22 else None,
            "stop_price":      float(parts[26]) if len(parts) > 26 else None,
        }
    else:
        raise ValueError(f"Nierozpoznany format CSV (brak 🔻 ani 🐚 w sygnale)")


def plot_candles(
    df: pd.DataFrame,
    symbol: str,
    start_str: str,
    end_str: str,
    markers: dict,
    hlines: list[dict],
    title: str,
) -> None:
    """Rysuje interaktywny wykres świecowy z wolumenem, markerami i poziomymi liniami."""

    colors = ["green" if c >= o else "red" for o, c in zip(df["Open"], df["Close"])]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # --- Wykres świecowy ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1, col=1,
    )

    # --- Poziome linie (confirmation, stop) ---
    for hline in hlines:
        fig.add_hline(
            y=hline["price"],
            line_dash=hline.get("dash", "dash"),
            line_color=hline.get("color", "white"),
            annotation_text=hline.get("label", ""),
            annotation_position="right",
            row=1, col=1,
        )

    # --- Markery formacji (pionowe daty) ---
    for date_str, cfg in markers.items():
        ts = pd.Timestamp(date_str)
        # Znajdź najbliższy dzień sesji
        idx = df.index.searchsorted(ts)
        if idx >= len(df.index):
            idx = len(df.index) - 1
        nearest = df.index[idx]
        price = cfg.get("price") or float(df.loc[nearest, "High"]) * 1.02
        fig.add_annotation(
            x=nearest,
            y=price,
            text=cfg["label"],
            showarrow=True,
            arrowhead=2,
            arrowcolor=cfg.get("color", "orange"),
            font=dict(size=10, color=cfg.get("color", "orange")),
            bgcolor="rgba(0,0,0,0.7)",
            row=1, col=1,
        )

    # --- Wolumen ---
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Wolumen",
                marker_color=colors,
                opacity=0.6,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{title}  |  {start_str} → {end_str}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Cena", row=1, col=1)
    fig.update_yaxes(title_text="Wolumen", row=2, col=1)

    fig.show()


def main() -> None:
    print("Wklej wiersz CSV z raportu i naciśnij Enter (lub wpisz 'q' aby wyjść):")
    line = sys.stdin.readline().strip()

    if line.lower() in ("q", "quit", "exit", ""):
        print("Wyjście.")
        return

    # --- Parsowanie wiersza CSV ---
    try:
        row = parse_csv_row(line)
    except (ValueError, IndexError) as e:
        print(f"Błąd parsowania wiersza: {e}")
        return

    ticker         = row["ticker"]
    breakout_date  = row["breakout_date"]
    breakout_price = row["breakout_price"]
    signal         = row["signal"]
    stop_price     = row.get("stop_price")
    fmt            = row.get("format", "double_bottom")

    if fmt == "scallop":
        a_date  = row["a_date"]
        c_date  = row["c_date"]
        b_date  = row["b_date"]
        a_price = row["a_price"]
        c_price = row["c_price"]
        b_price = row["b_price"]
        formation_start = datetime.strptime(a_date, "%Y-%m-%d")
        formation_end   = datetime.strptime(breakout_date, "%Y-%m-%d")
        formation_dates = {a_date, c_date, b_date, breakout_date}
        print(f"\nParsowano: {ticker}  |  {signal}")
        print(f"Formacja:  A={a_date} → C={c_date} → B={b_date}")
        print(f"Breakout:  {breakout_date}  @ {breakout_price:.3f}")
        markers = {
            a_date:        {"label": f"A (start) @ {a_price:.3f}", "color": "deepskyblue", "price": a_price * 0.97},
            c_date:        {"label": f"C (peak)  @ {c_price:.3f}", "color": "yellow",      "price": c_price * 1.02},
            b_date:        {"label": f"B (trough)@ {b_price:.3f}", "color": "orange",      "price": b_price * 0.97},
            breakout_date: {"label": f"BREAKOUT  @ {breakout_price:.3f}", "color": "springgreen", "price": breakout_price * 1.02},
        }
        hlines = [
            {"price": c_price, "label": f"Conf (C) {c_price:.3f}", "color": "lime",   "dash": "dash"},
        ]
        if stop_price:
            hlines.append({"price": stop_price, "label": f"Stop {stop_price:.3f}", "color": "tomato", "dash": "dot"})
    else:
        # double_bottom
        left_trough_date   = row["left_trough_date"]
        right_trough_date  = row["right_trough_date"]
        peak_date          = row["peak_date"]
        left_trough_price  = row["left_trough_price"]
        right_trough_price = row["right_trough_price"]
        peak_price         = row["peak_price"]
        confirmation_price = row["confirmation_price"]
        formation_start = datetime.strptime(left_trough_date, "%Y-%m-%d")
        formation_end   = datetime.strptime(right_trough_date, "%Y-%m-%d")
        formation_dates = {left_trough_date, right_trough_date, peak_date, breakout_date}
        print(f"\nParsowano: {ticker}  |  {signal}")
        print(f"Formacja:  {left_trough_date} → {right_trough_date}  (peak: {peak_date})")
        print(f"Breakout:  {breakout_date}  @ {breakout_price:.3f}")
        markers = {
            left_trough_date:  {"label": f"L1 @ {left_trough_price:.3f}",    "color": "cyan",        "price": left_trough_price  * 0.98},
            right_trough_date: {"label": f"L2 @ {right_trough_price:.3f}",   "color": "cyan",        "price": right_trough_price * 0.98},
            peak_date:         {"label": f"PEAK @ {peak_price:.3f}",          "color": "yellow",      "price": peak_price         * 1.02},
            breakout_date:     {"label": f"BREAKOUT @ {breakout_price:.3f}", "color": "springgreen", "price": breakout_price     * 1.02},
        }
        hlines = [
            {"price": confirmation_price, "label": f"Conf {confirmation_price:.3f}", "color": "lime",   "dash": "dash"},
        ]
        if stop_price:
            hlines.append({"price": stop_price, "label": f"Stop {stop_price:.3f}", "color": "tomato", "dash": "dot"})

    chart_start = formation_start - timedelta(days=MARGIN_DAYS)
    chart_end   = formation_end   + timedelta(days=MARGIN_DAYS)
    start_str = chart_start.strftime("%Y-%m-%d")
    end_str   = chart_end.strftime("%Y-%m-%d")

    symbol = ticker
    print(f"Wykres:    {start_str} → {end_str}  (margines ±{MARGIN_DAYS} dni)\n")
    print(f"Pobieram: {symbol} ...\n")

    df = yf.download(symbol, start=chart_start, end=chart_end + timedelta(days=1), auto_adjust=True, progress=False)

    if df is None or df.empty:
        print("Brak danych.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=-1)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna().sort_index()

    # --- Wypisz tabelę w terminalu ---
    print(f"{'Data':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 68)
    for date, r in df.iterrows():
        vol = f"{int(r['Volume']):,}" if "Volume" in r and pd.notna(r["Volume"]) else "n/a"
        tag = " ◄" if str(date.date()) in formation_dates else ""
        print(
            f"{str(date.date()):<12}"
            f" {float(r['Open']):>10.3f}"
            f" {float(r['High']):>10.3f}"
            f" {float(r['Low']):>10.3f}"
            f" {float(r['Close']):>10.3f}"
            f" {vol:>12}"
            f"{tag}"
        )

    print(f"\nŁącznie: {len(df)} sesji")

    # --- Wykres ---
    plot_candles(
        df=df,
        symbol=symbol,
        start_str=start_str,
        end_str=end_str,
        markers=markers,
        hlines=hlines,
        title=f"{symbol}  {signal}",
    )


if __name__ == "__main__":
    main()
