import feedparser
from datetime import date, datetime, timedelta, timezone
import time
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import hashlib
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket


PL_TZ = ZoneInfo("Europe/Warsaw")

FEEDS = [
    "https://www.bankier.pl/rss/gielda.xml",                 # Bankier - Giełda
    "https://www.bankier.pl/rss/espi.xml",                   # Bankier - ESPI
    "https://www.gpw.pl/rss/aktualnosci.xml",                # GPW - Aktualności
    "https://www.money.pl/rss/gielda",                       # Money.pl - Giełda (kanał RSS)
    "https://www.pb.pl/rss/puls-inwestora.xml",              # Puls Biznesu - Puls Inwestora
    "https://www.pb.pl/rss/notowania.xml",                   # Puls Biznesu - Notowania  (UWAGA: poprawiony URL)
]


COMPANY_KEYWORDS = {
    "INSIDER_TRADING": ["transakcj", "insider", "powiadomienie", "obowiązki zarządcze",
                        "obowiazki zarzadcze", "osoba", "19 mar", "19-mar", "mar 19", "mar-19",
                        "nabycie", "zbycie", "rozporządzenie", "rozporzadzenie"
                        ],
    "11BIT": ["11bit", "11"],
    "ACAUTOGAZ": ["autogaz"],
    "AILLERON": ["ailleron"],
    "ALLEGRO": ["allegro"],
    "AMICA": ["amica"],
    "AMREST": ["amrest"],
    "APATOR": ["apator"],
    "ARCTIC": ["arctic"],
    "ARLEN": ["arlen"],
    "ASSECOSEE/ASSECOPL": ["asseco"],
    "AUTOPARTN": ["auto", "partner"],
    "BIOCELTIX": ["bioceltix"],
    "BOGDANKA": ["bogdanka"],
    "BORYSZEW": ["boryszew"],
    "BOS": ["bank bos", "bank boś", "bank ochrony środowiska", "bank-bos", "bank-ochrony-srodowiska"],
    "BUDIMEX": ["budimex"],
    "BUMECH": ["bumech"],
    "CCC": ["ccc"],
    "CDPROJEKT": ["cdprojekt", "cd"],
    "CELON": ["celon"],
    "CIGAMES": ["ci games"], 
    "COGNOR": ["cognor"],
    "CREOTECH": ["creotech"],
    "CYFROWYPOLSAT": ["polsat"],
    "DATAWALK": ["datawalk"],
    "DIAG": ["diag"],
    "DINOPL": ["dino"],
    "ELEKTROTI": ["elektroti"],
    "ENTER": ["enter"],
    "ERBUD": ["erbud"],
    "EUROCASH": ["eurocash"],
    "FERRO": ["ferro"],
    "GRENEVIA": ["grenevia"],
    "GRUPAAZOTY": ["azoty"],
    "GRUPRACUJ": ["pracuj"],
    "HUUUGE": ["huuuge"],
    "KGHM": ["kghm"],
    "LUBAWA": ["lubawa"],
    "MABION": ["mabion"],
    "MCR": ["mcr"],
    "MEDICALG": ["medicalg"],
    "MENNICA": ["mennica"],
    "MERCATOR": ["mercator"],
    "MLSYSTEM": ["mlsystem", "ml-system", "ml system"],
    "MURAPOL": ["murapol"],
    "ONDE": ["onde"],
    "PCCROKITA": ["rokita"],
    "PEKABEX": ["pekabex"],
    "PEPCO": ["pepco"],
    "PGE": ["pge"],
    "PKNORLEN": ["pkn", "orlen"],
    "PLAYWAY": ["playway"],
    "POLIMEXMS": ["polimex"],
    "QUERCUS": ["quercus"],
    "RYVU": ["ryvu"],
    "SELVITA": ["selvita"],
    "SHOPER": ["shoper"],
    "SNTVERSE": ["synthaverse", "sntverse"],
    "STALEXP": ["stalexp", "autostrad"],
    "STALEXPORT": ["stalexport"],
    "TARCZYNSKI": ["tarczynski", "tarczyński"],
    "TEXT": ["text"],
    "TORPOL": ["torpol"],
    "TSGAMES": ["ts games", "games", "ten square"],
    "UNIMOT": ["unimot"],
    "VOTUM": ["votum"],
    "VOXEL": ["voxel"],
    "WIELTON": ["wielton"],
    "WIRTUALNA": ["wirtualna"], 
    "WITTCHEN": ["wittchen"],
    "XTB": ["xtb"],
    "ZEPAK": ["zepak", "ze-pak", "ze pak"]
}


def send_email(body: str, body_html = None):
    msg = MIMEMultipart("alternative") if body_html else MIMEMultipart()
    msg["From"] = "kielarzu@gmail.com"
    msg["To"] = "kielarzu@gmail.com"
    msg["Subject"] = "Raport RSS z " + date.today().strftime("%d.%m.%Y")
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


def to_dt_utc(struct_time_obj):
    """Konwersja feedparser struct_time -> datetime (UTC)."""
    if not struct_time_obj:
        return None
    return datetime.fromtimestamp(time.mktime(struct_time_obj), tz=timezone.utc)

def entry_dt_utc(entry):
    """Bierz published_parsed lub updated_parsed (UTC)."""
    return to_dt_utc(entry.get("published_parsed")) or to_dt_utc(entry.get("updated_parsed"))

def window_prev_day_6am_pl():
    """Okno: od 06:00 dnia poprzedniego (PL) do teraz."""
    now_pl = datetime.now(PL_TZ)
    prev_pl_date = (now_pl - timedelta(days=1)).date()
    start_pl = datetime(prev_pl_date.year, prev_pl_date.month, prev_pl_date.day, 6, 0, tzinfo=PL_TZ)
    end_pl = now_pl
    return start_pl.astimezone(timezone.utc), end_pl.astimezone(timezone.utc), start_pl, end_pl

def source_tag(url, feed_title=None):
    """Zwróć krótki tag źródła na podstawie domeny lub tytułu kanału."""
    netloc = urlparse(url).netloc.lower()
    if "bankier.pl" in netloc:
        return "Bankier"
    if "gpw.pl" in netloc:
        return "GPW"
    if "money.pl" in netloc:
        return "Money.pl"
    if "pb.pl" in netloc:
        return "PB"
    # fallback na tytuł lub domenę
    return (feed_title or netloc).split()[0]

def guid(title, link):
    """Hash do deduplikacji wpisów."""
    base = (title or "") + "|" + (link or "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def main():
    start_utc, end_utc, start_pl, end_pl = window_prev_day_6am_pl()
    report = f"Okno: od [{start_pl:%Y-%m-%d %H:%M} PL] do [{end_pl:%Y-%m-%d %H:%M} PL]\n\n"

    all_items = []
    seen = set()
    per_source_counts = {}

    for rss_url in FEEDS:
        try:
            feed = feedparser.parse(rss_url)
        except Exception as ex:
            report = report + f"[WARN] Nie udało się pobrać: {rss_url} -> {ex}"
            continue

        src = source_tag(rss_url, feed.feed.get("title"))
        count_selected = 0

        for e in feed.entries:
            d_utc = entry_dt_utc(e)
            if not d_utc or not (start_utc <= d_utc <= end_utc):
                continue

            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            g = guid(title, link)
            if g in seen:
                continue

            seen.add(g)
            count_selected += 1
            all_items.append((d_utc, src, title, link))

        per_source_counts[src] = per_source_counts.get(src, 0) + count_selected

    # sort globalnie malejąco po czasie
    all_items.sort(key=lambda x: x[0], reverse=True)

    # wypisz podsumowanie per źródło
    # if per_source_counts:
    #     report = report + ("Podsumowanie per źródło (liczba wpisów w oknie):")
    #     for src, cnt in sorted(per_source_counts.items(), key=lambda x: x[0]):
    #         report = report + (f"- {src}: {cnt}")
    #     report = report + ("")

    # wypisz szczegóły
    if not all_items:
        report = report + ("Brak wpisów w zadanym oknie czasu.") 
        return

    
    ALL_TOKENS = {tok.lower() for toks in COMPANY_KEYWORDS.values() for tok in toks}

    report = report + ("Newsy:\n\n")
    for d_utc, src, title, link in all_items:
        t = (title or "").lower()
        l = (link or "").lower()

        if any(tok in t for tok in ALL_TOKENS):
            d_pl = d_utc.astimezone(PL_TZ)
            report = report + (f"[{d_pl:%Y-%m-%d %H:%M} PL] [{src}] {title}") + "\n"

            report = report + ("-" * 90) + "\n"

    report = report + ("\n\nLinki:\n") 
    for d_utc, src, title, link in all_items:
        t = (title or "").lower()
        l = (link or "").lower()

        if any(tok in t for tok in ALL_TOKENS):
            d_pl = d_utc.astimezone(PL_TZ)
            report = report + (link) + "\n\n"

    print(report)
    send_email(report)

if __name__ == "__main__":
    main()
