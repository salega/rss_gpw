import feedparser
from datetime import date, datetime, timedelta, timezone
import time
from itertools import islice
from zoneinfo import ZoneInfo
import hashlib
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

from espi import get_espi_links_for_company
from data import COMPANY_KEYWORDS

PL_TZ = ZoneInfo("Europe/Warsaw")

FEEDS = [
    "https://www.bankier.pl/rss/gielda.xml",                 # Bankier - Giełda
    "https://www.bankier.pl/rss/espi.xml",                   # Bankier - ESPI
    "https://www.money.pl/rss/gielda",                       # Money.pl - Giełda (kanał RSS)
    "https://www.pb.pl/rss/puls-inwestora.xml",              # Puls Biznesu - Puls Inwestora
    "https://www.pb.pl/rss/notowania.xml",                   # Puls Biznesu - Notowania  (UWAGA: poprawiony URL)
]


def send_email(body: str, body_html=None):
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
    timeout_sec = 5

    last_err = None
    for i in range(1, attempts + 1):
        try:
            with smtplib.SMTP_SSL(host, port, timeout=timeout_sec) as server:
                server.login(username, smtp_password)
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())
            return
        except (smtplib.SMTPException, socket.timeout, OSError) as e:
            last_err = e
            sleep_s = min(60, base_backoff * (2 ** (i - 1)))
            time.sleep(sleep_s)
            continue
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"Nie udało się wysłać maila po {attempts} próbach: {last_err}")


def to_dt_utc(struct_time_obj):
    if not struct_time_obj:
        return None
    return datetime.fromtimestamp(time.mktime(struct_time_obj), tz=timezone.utc)


def entry_dt_utc(entry):
    return to_dt_utc(entry.get("published_parsed")) or to_dt_utc(entry.get("updated_parsed"))


def window_prev_day_6am_pl():
    now_pl = datetime.now(PL_TZ)
    prev_pl_date = (now_pl - timedelta(days=1)).date()
    start_pl = datetime(prev_pl_date.year, prev_pl_date.month, prev_pl_date.day, 6, 0, tzinfo=PL_TZ)
    end_pl = now_pl
    return start_pl.astimezone(timezone.utc), end_pl.astimezone(timezone.utc), start_pl, end_pl


def guid(title, link):
    base = (title or "") + "|" + (link or "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def render_item(time_text: str, title_html: str, link_url: str) -> str:
    link = f'[<a href="{link_url}">link</a>]\n'
    return f'<span style="font-size: 0.8em;">⏰{time_text} - <b>{title_html}</b> {link}</span>\n'


def collect_rss_items(feeds, start_utc, end_utc):
    all_items = []
    seen = set()
    errors = ""

    for rss_url in feeds:
        try:
            feed = feedparser.parse(rss_url)
        except Exception as ex:
            errors = errors + f"[WARN] Nie udało się pobrać: {rss_url} -> {ex}"
            continue

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
            all_items.append((d_utc, title, link))

    all_items.sort(key=lambda x: x[0], reverse=True)
    return all_items, errors


def collect_espi_links_for_all_companies(company_keywords: dict):
    all_espi_links = []
    seen = set()

    for company_name, keywords in company_keywords.items():
        try:
            links = get_espi_links_for_company("GPW", keywords, 1)
            for link in links:
                g = guid(link[1], link[0])
                if g in seen:
                    continue

                all_espi_links.append((link[0], link[1], link[2], company_name))
                seen.add(g)
        except Exception as ex:
            all_espi_links.append(("", f"ESPI error: {ex}", "", company_name))

    return all_espi_links


def main():
    start_utc, end_utc, start_pl, end_pl = window_prev_day_6am_pl()
    report = f"<b><span style='font-size: 1.25em;'>Newsy dla:\n📅{start_pl:%Y-%m-%d %H:%M} -\n📅{end_pl:%Y-%m-%d %H:%M}</span></b>\n\n"

    all_items, errors = collect_rss_items(FEEDS, start_utc, end_utc)

    all_tokens = {token.lower() for tokens in COMPANY_KEYWORDS.values() for token in tokens}

    for d_utc, title, link in all_items:
        t = (title or "").lower()
        if any(tok in t for tok in all_tokens):
            d_pl = d_utc.astimezone(PL_TZ)
            report = report + render_item(f"{d_pl:%H:%M}", title, link)

    if not all_items:
        report = report + "Brak wpisów w zadanym oknie czasu."

    espi_links = collect_espi_links_for_all_companies(dict(islice(COMPANY_KEYWORDS.items(), 1, None)))
    if espi_links:
        report = report + '<br><br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br><br>'
        for espi_link, title, time_str, company in espi_links:
            report = report + render_item(time_str, f"{company}: {title}", espi_link)

    if errors:
        report = report + "\n" + errors

    print(report)
    send_email(
        report,
        body_html=f"""\
        <html>
          <body>
            <pre style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 
            'Courier New', monospace;">{report}</pre>
          </body>
        </html>
        """)


if __name__ == "__main__":
    main()