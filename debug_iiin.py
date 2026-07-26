"""
Debug script — uruchom: python debug_iiin.py
Sprawdza krok po kroku dlaczego IIIN nie jest wykrywane.
"""
import io
import pandas as pd
from formations.flag import _prepare_flag_df, _find_pole_from_index, _find_breakout_after_pole

CSV = """Date,Open,High,Low,Close,Volume
2005-12-13,5.60685,5.80876,5.54406,5.80876,95595.576271044
2005-12-14,5.76028,5.76028,5.57932,5.69721,52874.106485999
2005-12-15,5.58584,5.71144,5.537,5.68353,40207.4937906
2005-12-16,5.70437,5.85072,5.61375,5.63468,98688.873796489
2005-12-19,5.66967,5.76708,5.53029,5.53029,24061.727485101
2005-12-20,5.55114,5.63468,5.55114,5.55114,35045.190797302
2005-12-21,5.53029,5.64175,5.53029,5.57216,23126.644494435
2005-12-22,5.67664,5.72522,5.50945,5.50945,55451.496154525
2005-12-23,5.59998,5.69032,5.57216,5.66967,12892.322382802
2005-12-27,5.72522,5.76028,5.48163,5.57932,49055.313536763
2005-12-28,5.537,5.59998,5.43948,5.44673,37800.998599679
2005-12-29,5.46042,5.73898,5.46042,5.68353,44236.949114745
2005-12-30,5.56499,5.63468,5.48842,5.58584,35795.406805906
2006-01-03,5.48163,5.73898,5.48163,5.73898,70147.346420473
2006-01-04,5.80196,5.93436,5.75312,5.86458,176342.67953513
2006-01-05,5.93436,5.93436,5.80196,5.82271,45484.801243668
2006-01-06,5.93436,5.94822,5.80196,5.94822,131294.25035378
2006-01-09,5.99688,6.09429,5.8858,6.09429,188072.05962379
2006-01-10,6.19217,6.19217,5.95483,6.05262,99834.619162063
2006-01-11,6.15692,6.15692,6.07345,6.0876,86302.785998289
2006-01-12,6.15692,6.15692,6.04583,6.10825,103692.10520057
2006-01-13,6.06666,6.7212,6.06666,6.65858,543402.52126882
2006-01-17,6.65858,6.93733,6.65858,6.92356,355643.23078328
2006-01-18,6.78392,6.93733,6.57503,6.70743,123567.45538838
2006-01-19,6.81182,6.87556,6.66556,6.75609,164828.26105352
2006-01-20,6.81879,7.02768,6.76297,6.88867,116728.4518601
2006-01-23,6.95818,8.05102,6.93733,7.98952,1089888.6667907
2006-01-24,7.98952,8.55455,7.95472,8.52627,762594.57274518
2006-01-25,8.34364,8.574,8.26734,8.35768,365669.04013607
2006-01-26,8.35768,8.44132,8.35768,8.36522,207198.26861105
2006-01-27,8.42718,8.6581,8.38634,8.45453,338044.32401411
2006-01-30,8.64396,8.79617,8.46867,8.76872,320619.53614667
2006-01-31,8.79617,9.40077,8.76073,9.23514,347271.55099673
2006-02-01,9.22118,9.33189,9.09651,9.21467,229382.30645856
2006-02-02,9.14303,9.35426,9.0351,9.33189,220448.50206958
2006-02-03,9.35426,10.0241,9.19327,9.77015,729836.5734469
2006-02-06,9.83987,9.97474,9.63798,9.81946,364839.28833286
2006-02-07,9.90502,9.96271,9.40077,9.86876,782667.6876115
2006-02-08,9.86876,10.1134,9.38213,9.74405,541489.36296607
2006-02-09,9.6734,9.91247,9.47422,9.81946,472598.46713882
2006-02-10,9.77015,10.1851,9.77015,9.99617,365376.69235049
2006-02-13,10.1404,10.748,10.0501,10.5097,718422.11211256
2006-02-14,10.4549,10.601,9.98872,10.5377,349710.2904287
2006-02-15,10.628,10.7536,10.573,10.6977,350621.7276426
2006-02-16,10.6977,10.8019,10.5377,10.734,267778.74870977
2006-02-17,10.7814,10.7814,10.6633,10.7415,377508.05064433
2006-02-21,10.8158,11.1369,10.4475,11.0541,515323.16134796
2006-02-22,11.1713,11.4384,11.0541,11.4085,304142.72896633
2006-02-23,11.4514,11.7835,11.2932,11.7267,479763.13750177
2006-02-24,11.776,11.91,11.6923,11.8263,368684.95148298
2006-02-27,11.9668,12.5176,11.9017,12.2803,1152024.3941165
2006-02-28,12.2999,12.5855,12.1901,12.5781,694640.90952466
2006-03-01,12.8721,13.3094,12.6479,13.2452,1124326.5910482
2006-03-02,13.394,13.7309,12.9763,13.675,1690547.2869054
2006-03-03,13.6247,14.089,12.8926,14.0063,1749695.0478931
2006-03-06,14.0184,14.2714,13.7773,14.1095,1411296.0372272
2006-03-07,14.1095,14.1523,13.5671,13.8629,1584023.0625318
2006-03-08,13.4145,14.0965,12.9837,13.702,1381492.6852179
2006-03-09,13.7699,14.1886,13.7178,14.0184,874622.88906714
2006-03-10,14.1886,14.5421,14.1523,14.3737,1059528.5642191
2006-03-13,14.5338,14.891,14.41,14.8491,755678.18303837
2006-03-14,14.9952,15.3516,14.959,15.3097,1228405.6271414
2006-03-15,15.3563,16.2792,15.3563,16.175,1305385.528242
2006-03-16,16.3396,16.8021,14.3449,14.6622,4539585.0130671
2006-03-17,14.6007,15.4678,14.3533,15.3563,1584814.1212458
2006-03-20,15.4213,15.7684,14.8418,15.5795,1379772.9923615
2006-03-21,15.7349,15.8372,15.3097,15.4762,794785.07340088
2006-03-22,15.5033,16.7676,15.5033,16.5127,1747757.1690055
2006-03-23,16.6597,17.1147,16.175,17.0719,1498473.7169662
2006-03-24,17.2273,17.9139,17.2189,17.845,1290230.734945
"""

df = pd.read_csv(io.StringIO(CSV), index_col="Date", parse_dates=True)
working_df = _prepare_flag_df(df)

print(f"Liczba wierszy: {len(working_df)}")
print(f"Kolumny: {list(working_df.columns)}")
print()

POLE_MIN_GROWTH = 0.85
POLE_MAX_DAILY_DECLINE = 0.20
POLE_MIN_DAYS = 4
POLE_MAX_DAYS = 40
MAX_DAYS_WITHOUT_NEW_HIGH = 3

# Szukamy masztu startującego od 2006-01-13
target_date = pd.Timestamp("2006-01-13")
start_idx = working_df.index.get_loc(target_date)
print(f"Start masztu: idx={start_idx}, date={working_df.index[start_idx]}, Close={working_df.iloc[start_idx]['Close']:.2f}")
print()

# Ręczna symulacja pętli masztu
pole_start_close = float(working_df.iloc[start_idx]["Close"])
current_max = float(working_df.iloc[start_idx]["High"])
actual_pole_end_idx = start_idx
closes = []
max_end_idx = min(len(working_df) - 1, start_idx + POLE_MAX_DAYS - 1)

print(f"=== Symulacja pętli masztu (max_days_without_new_high={MAX_DAYS_WITHOUT_NEW_HIGH}) ===")
days_without_new_high_sim = 0
for idx in range(start_idx, max_end_idx + 1):
    row = working_df.iloc[idx]
    current_close = float(row["Close"])
    current_high = float(row["High"])
    closes.append(current_close)
    local_i = idx - start_idx

    if current_high > current_max:
        current_max = current_high
        actual_pole_end_idx = idx
        days_without_new_high_sim = 0
    else:
        days_without_new_high_sim += 1

    stopped = days_without_new_high_sim >= MAX_DAYS_WITHOUT_NEW_HIGH
    rejected = current_close < pole_start_close

    print(
        f"  {working_df.index[idx].date()} | Close={current_close:.2f} | High={current_high:.2f} | "
        f"current_max={current_max:.2f} | no_high_days={days_without_new_high_sim} | "
        f"{'⛔ STOP (no new high)' if stopped else ''}{'❌ ODRZUCONO (close < start)' if rejected else ''}"
    )
    if stopped or rejected:
        break

print()
pole_growth = (current_max - pole_start_close) / pole_start_close
print(f"Szczyt masztu: {working_df.index[actual_pole_end_idx].date()}, High={current_max:.2f}")
print(f"Wzrost masztu: {pole_growth * 100:.1f}% (min wymagane: {POLE_MIN_GROWTH * 100:.0f}%)")
print(f"Długość masztu: {actual_pole_end_idx - start_idx + 1} dni (min: {POLE_MIN_DAYS})")
print()

# Teraz sprawdź przez właściwą funkcję
pole = _find_pole_from_index(
    working_df=working_df,
    start_idx=start_idx,
    pole_min_days=POLE_MIN_DAYS,
    pole_max_days=POLE_MAX_DAYS,
    pole_min_growth=POLE_MIN_GROWTH,
    pole_max_daily_decline=POLE_MAX_DAILY_DECLINE,
    max_days_without_new_high=MAX_DAYS_WITHOUT_NEW_HIGH,
)

if pole is None:
    print("❌ _find_pole_from_index zwróciło None")
else:
    print(f"✅ Maszt znaleziony: {pole['pole_start_date'].date()} → {pole['pole_peak_date'].date()}, "
          f"wzrost={pole['pole_growth']*100:.1f}%, max_price={pole['max_price']:.2f}")
    breakout = _find_breakout_after_pole(
        working_df=working_df,
        pole=pole,
        flag_min_days=3,
        flag_max_days_until_breakout=19,
        require_volume_decline=False,
        require_dense_flag=False,
    )
    if breakout is None:
        print("❌ _find_breakout_after_pole zwróciło None")
        flag_start_idx = pole["pole_end_idx"] + 1
        print(f"   flag_start_idx={flag_start_idx}, date={working_df.index[flag_start_idx].date()}")
        print(f"   max_price (szczyt masztu High)={pole['max_price']:.2f}")
        print(f"   half_pole={pole['pole_start_price'] + pole['pole_height'] / 2:.2f}")
        print(f"   Sesje dostępne po szczycie: {len(working_df) - flag_start_idx}")
        print()
        print("=== Symulacja flagi ===")
        flag_candle_highs = []
        flag_candle_lows = []
        max_price = pole["max_price"]
        pole_height = pole["pole_height"]
        half_pole = pole["pole_start_price"] + pole_height / 2.0
        FLAG_MAX_DAYS = 19
        for idx in range(flag_start_idx, min(len(working_df), flag_start_idx + FLAG_MAX_DAYS + 1)):
            row = working_df.iloc[idx]
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            days_in_flag = idx - flag_start_idx + 1
            flag_high_so_far = max(flag_candle_highs) if flag_candle_highs else 0.0
            retracement = (max_price - min(flag_candle_lows)) / pole_height if flag_candle_lows else 0.0

            breakout_triggered = close_price > flag_high_so_far and flag_high_so_far > 0 and days_in_flag > 1
            below_half = low_price < half_pole
            retr_exceeded = retracement > 0.25

            print(
                f"  {working_df.index[idx].date()} | Close={close_price:.2f} | High={high_price:.2f} | Low={low_price:.2f} | "
                f"flag_high={flag_high_so_far:.2f} | retracement={retracement*100:.1f}% | "
                f"{'🔔 BREAKOUT' if breakout_triggered else ''}"
                f"{'❌ BELOW_HALF' if below_half else ''}"
                f"{'❌ RETR>25%' if retr_exceeded else ''}"
            )
            flag_candle_highs.append(high_price)
            flag_candle_lows.append(low_price)
            if below_half or retr_exceeded:
                print("   → pętla przerwana")
                break
    else:
        print(f"✅ Breakout znaleziony: {breakout['breakout_date'].date()}")
