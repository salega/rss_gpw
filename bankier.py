# Co analizować:
# 1. Fale Elliotta
# 2. Linia oporu i wsparcia + EMA
# 3. MACD
# 4. RSI i Oscylator stochastyczny
# 5. Analiza fundamentalna: dla danej spółki wyszukaj linki do newsów + linki do ESPI z ostatnich X dni.
import os

import requests
import time
import random
from bs4 import BeautifulSoup
import sys
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import pandas as pd

from espi import get_espi_links_for_company
from data import WIG_SECTOR_BY_TICKER

warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_URL = "https://www.bankier.pl"
NEWS_URL = f"{BASE_URL}/gielda/wiadomosci/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MOCKING_PERPLEXITY_ENABLED = False
MOCKING_NEWS_ENABLED = False
NEWS_MAX_PAGE = 50
STOCK_EXCHANGES = {
    'GPW': '.WA', # Polska -> Giełda Papierów Wartościowych w Warszawie
    'NYSE': ''    # USA -> New York Stock Exchange
}


def ask_perplexity_api(company: str, question: str, company_document_links = []):
    endpoint = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": os.environ["PERPLEXITY_TOKEN"],
        "Content-Type": "application/json"
    }

    user_content = [
        {
            "type": "text",
            "text": question
        }
    ]

    for u in company_document_links:
        user_content.append({
            "type": "file_url",
            "file_url": {
                "url": u.strip()
            }
        })

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Jesteś analitykiem finansowym. Bądź precyzyjny, rzetelny, szukaj dodatkowych informacji dla deep research."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

    if MOCKING_PERPLEXITY_ENABLED:
        return "<<< Mock Perplexity response for query: \n\n " + question + "\n\n>>>"
        #return "<<< Mock Perplexity response for query: \n\n " + json.dumps(payload) + "\n\n>>>"

    response = requests.post(endpoint, json=payload, headers=headers)

    if response.status_code in (401, 403):
        return "Nie masz już możliwości callowania Perplexity, doładuj konto."

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_report = question + "\n\n\ncompany_document_links = \n" + str(company_document_links)
        save_report_to_file(company, error_report, True)
        return f"Wystąpił błąd: {e}"

    data = response.json()
    return data["choices"][0]["message"]["content"]


def sector_2_years_data(company_abbr: str):
    sector = WIG_SECTOR_BY_TICKER.get(company_abbr)
    if not sector:
        return "Spółka nie należy do żadnego indeksu sektorowego, brak danych."

    s = sector.lower().replace("-", "_")
    today = datetime.today()
    start = today - timedelta(days=730)

    f = start.strftime("%Y%m%d")
    t = today.strftime("%Y%m%d")

    url = f"https://stooq.pl/q/d/l/?s={s}&f={f}&t={t}&i=wg"

    try:
        df = pd.read_csv(url)
    except Exception:
        return ""

    if df.empty or "Data" not in df.columns or "Zamkniecie" not in df.columns:
        return ""

    df = df[["Data", "Zamkniecie"]].dropna()
    df["Data"] = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m-%d")

    return "\n".join(f"{d}, {c}" for d, c in zip(df["Data"], df["Zamkniecie"]))


def get_4_year_price_data(company_abbr: str, stock_exchange: str):
    data = yf.download(company_abbr + STOCK_EXCHANGES[stock_exchange], period="4y", interval="1wk", auto_adjust=False, progress=False)
    price = data[['Close']]

    return price.to_string()


def get_half_year_price_data(company_abbr: str, stock_exchange: str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    data = yf.download(company_abbr + STOCK_EXCHANGES[stock_exchange], start=start_date, end=end_date, progress=False)
    price = data[['Close']]

    return price.to_string()


def get_last_week_price_data(company_abbr: str, stock_exchange: str):
    data = yf.download(company_abbr + STOCK_EXCHANGES[stock_exchange], period="7d", interval="1h", auto_adjust=False, progress=False)
    price = data[['Close']]
    return price.to_string()


def contains_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    # uncomment to see the response
    # print(resp.text)
    return BeautifulSoup(resp.text, "html.parser")


def news_is_older_than(news_date, date_to_compare):
    format_str = "%Y-%m-%d %H:%M"
    date1 = datetime.strptime(news_date, format_str)
    date2 = datetime.strptime(date_to_compare, format_str)
    return date2 > date1


def get_news_links_for_page(page_number: str, company_keywords, date_to_compare: str):
    all_news_links_per_page = []
    try:
        soup_main = get_soup(NEWS_URL + page_number)
        news_links = soup_main.select("li.m-listing-article-list__item a.m-listing-article-list__anchor")
        
        for a_tag in news_links:
            news_date = a_tag.select_one("div.m-listing-article-list__date-time").get_text(strip=True)
            if news_date == 'Reklama':
                continue
            if news_is_older_than(news_date, date_to_compare):
                return all_news_links_per_page

            href = a_tag.get("href", "")
            title = a_tag.select_one("div.m-listing-article-list__title").get_text(strip=True)
            link = href if href.startswith("http") else BASE_URL + href

            if contains_keyword(title, company_keywords) or contains_keyword(link, company_keywords):
                all_news_links_per_page.append(link)
 
        
    except Exception as ex:
        print(f"get_news_links_for_page: [Błąd pobierania strony z artykułami: {ex}]")
        
    return all_news_links_per_page


def get_date_30_days_ago():
    target_day = datetime.now() - timedelta(days=30)
    return target_day.strftime("%Y-%m-%d %H:%M")


def analyze_company(company: str, company_abbr: str, stock_exchange: str, company_keywords = [], company_document_links = [], sector_data = ""):
    news_links = get_news_links_for_company(stock_exchange, company_keywords)
    espi_links = [link for link, *_ in get_espi_links_for_company(stock_exchange, company_keywords)]
    all_company_links = news_links + espi_links + company_document_links

    company_links_query = "Do zapytania załączyłem linki do newsów, newsów ESPI, raportów finansowych i innych oficjalnych dokumentów spółki " + company + "\n\n"

    three_years_price = get_4_year_price_data(company_abbr, stock_exchange)
    three_years_price_query = "Oto dane historyczne spółki z ostatnich 3 lat (ceny zamknięcia dla każdego tygodnia) dla " + company + ":\n\n" + three_years_price + "\n\n"

    half_year_price = get_half_year_price_data(company_abbr, stock_exchange)
    half_year_price_query = "Oto dane historyczne spółki z ostatnich 6 miesięcy (ceny zamknięcia dla każdego dnia) dla " + company + ":\n\n" + half_year_price + "\n\n"

    last_week_price = get_last_week_price_data(company_abbr, stock_exchange)
    last_week_price_query = "Oto dane historyczne spółki z ostatniego tygodnia (ceny zamknięcia dla każdej godziny) dla " + company + ":\n\n" + last_week_price + "\n\n"

    sector = WIG_SECTOR_BY_TICKER.get(company_abbr, "")
    print("\nSektor dla tej spółki: " + sector)
    sector_price_query = "Oto dane historyczne indeksu sektorowego " + sector + " (ceny zamknięcia dla każdego tygodnia):\n" + sector_data + "\n\n" if sector else ""

    intro = "Interesuje mnie analiza finansowa/ekonomiczna dla spółki o nazwie '" + company + "' notowanej na " + stock_exchange + ". Poniżej zamieszczam niezbędne dane, a na samym dole jest moje zapytanie.\n\n"
    full_query = intro + company_links_query + three_years_price_query + half_year_price_query + last_week_price_query + sector_price_query + prepare_request(bool(company_document_links))
    report = "\n\n ==================================================== \n\n" + ask_perplexity_api(company, full_query, all_company_links) + "\n\n ==================================================== \n\n"
    return report


def get_news_links_for_company(stock_exchange: str, company_keywords):
    if stock_exchange != 'GPW':
        return []

    # ---- TEST ----
    if MOCKING_NEWS_ENABLED:
        return [ "https://www.bankier.pl/wiadomosc/DM-BOS-obnizyl-wycene-akcji-JSW-do-16-zl-8964024.html",
                "https://www.bankier.pl/wiadomosc/DM-BOS-obnizyl-wycene-akcji-JSW-do-16-zl-8964024.html" ]
    # ---- TEST ----

    all_news_links = []
    for i in range(1, NEWS_MAX_PAGE + 1):
        print("Pobieranie newsów ze strony " + str(i))
        all_news_links = all_news_links + get_news_links_for_page(str(i), company_keywords, get_date_30_days_ago())
        time.sleep(random.uniform(0.5, 1))

    print("\nLinki do newsów:")
    for news_link in all_news_links:
        print(news_link)
    print("")

    return all_news_links


def prepare_request(has_reports: bool):
    find_reports_query = "Przeanalizuj załączone dokumenty dot. spółki i podsumuj w kilku zdaniach te dokumenty prostym językiem, a także uwzględnij je do późniejszej predykcji notowań " \
        if has_reports else "Znajdź ostatni raport roczny + 2 raporty kwartalne, prezentacje wyników z ostatnich 4 kwartałów (jeśli są), MD&A oraz wszelkie oficjalne raporty ze strony spółki i przeanalizuj je"
    return (
    "\n\nMoje zapytanie: " + find_reports_query + ", a także przeanalizuj załączone dane historyczne ceny akcji, notowania indeksu sektorowego,"
    " newsy dotyczące tej spółki i oceń szanse na wzrost lub spadek wartości akcji dla swing tradera."
    " Dodatkowo dodaj bardzo krótką analizę sektorową (1 zdaniem) - jeśli załączono notowania indeksu sektorowego, do którego należy ta spółka, to oprzyj się na tym i sprawdź jak spółka "
    " radzi sobie w ostatnim czasie i ostatnim roku w stosunku do sektora oraz jak bardzo ich notowania są skorelowane z sektorem."
    " Sprawdź, co może wpływać na notowania spółki (np. sprawdź ceny miedzi jeśli analizowany jest KGHM, państwowe akcje profilaktyczne jeśli analizowana jest Diagnostyka itd.) "
    " oraz czy jakieś ostatnie wydarzenia mogą mieć wpływ na spółkę lub czy notowania spółki zależą od jakiegoś zdarzenia, które za niedługo może"
    " się stać np. nowy kontrakt, nowy produkt lub nowe przepisy. Napisz od czego głównie zależą zyski tej spółki i jak te czynniki obecnie"
    " wpływają na tę spółkę. Wyszukaj dodatkowe dane, które uznasz za cenne, poza tymi, które załączyłem, aby zrobić deep research. "
    " Dodatkowo wykonaj bardzo krótką analizę makroekonomiczną."
    " Poza tym pokaż wartości najważniejszych wskaźników analizy fundamentalnej (najlepiej na podstawie załączonych dokumentów i stron) tzn.: "
    " podsumowanie rachunku zysków i strat, EBITDA, przychody, koszty "
    " (i pokaż jak zmieniały się przychody, EBITDA i koszty z raportu na raport, chcę widzieć, czy przychody i koszty rosną, czy maleją), cena/wartość księgowa, zadłużenie"
    " , płynność, rentowność oraz generalnie oceń czy wartość akcji w tej chwili jest raczej tania, czy droga. Podsumuj jednym zdaniem tę część analizy fundamentalnej."
    " Opisz wartości wskaźników bardzo zwięźle, tj. oceń czy wskaźnik ma wartość korzystną, niekorzystną, czy coś pomiędzy. Przetłumacz też wskaźniki na język polski."
    " Zwróć też uwagę, czy wykres nie tworzy fal Elliotta."
    " Na koniec oceń na podstawie wszystkich załączonych danych (dokumentów, newsów, notowań, własnych analiz), czy wartość akcji tej spółki urośnie, "
    " czy spadnie oraz w jakim terminie i dlaczego. Interesuje mnie predykcja pod kątem swing tradingu lub pod kątem długiej inwestycji np. wielomiesięcznej lub rocznej."
    " Dopisz, czy Twoim zdaniem większa jest szansa na wzrost, czy spadek wartości akcji oraz procent od 0 do 100, który ocenia Twoim zdaniem prawdopodobieństwo na to zdarzenie w najbliższej przyszłości"
    " (i czy to raczej szybko, czy w dłuższym czasie). W raporcie rozwijaj skróty i rób krótkie przypisy, jeśli jakiś skrót może wymagać branżowej wiedzy."
    " Na samym dole odpowiedzi podaj wszystkie linki do źródeł, których użyłeś (te moje i te, które sam znalazłeś)."
    )


def fetch_order_books(company: str):
    url = f"https://gragieldowa.pl/spolka_arkusz_zl/spolka/{company}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    def parse_table_by_id(table_id):
        table = soup.find("table", id=table_id)
        orders = []
        if not table:
            return orders

        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = [col.get_text(strip=True).replace(',', '.').replace('\xa0', '') for col in row.find_all("td")]
            if len(cols) >= 5:
                try:
                    order = {
                        "price": float(cols[0]),
                        "volume": int(cols[1].replace(' ', '')),
                        "value": float(cols[2].replace(' ', '')),
                        "count": int(cols[3]),
                        "percent": cols[4]
                    }
                    orders.append(order)
                except ValueError:
                    continue
        return orders

    buy_orders = parse_table_by_id("arkusz_left")
    sell_orders = parse_table_by_id("arkusz_right")
    return buy_orders, sell_orders


def calculate_opening_price(buy_orders, sell_orders):
    if not buy_orders or not sell_orders:
        print("Brak danych w arkuszu kupna lub sprzedaży.")
        return -1

    buy_sorted = sorted(buy_orders, key=lambda x: -x["price"])
    sell_sorted = sorted(sell_orders, key=lambda x: x["price"])

    possible_prices = set([o["price"] for o in buy_sorted]) & set([o["price"] for o in sell_sorted])
    if not possible_prices:
        print("Brak przecięcia cen kupna i sprzedaży — transakcja nie może zostać zawarta.")
        return -1

    best_price = None
    max_volume = 0

    for price in sorted(possible_prices):
        buy_volume = sum(o["volume"] for o in buy_sorted if o["price"] >= price)
        sell_volume = sum(o["volume"] for o in sell_sorted if o["price"] <= price)
        matched_volume = min(buy_volume, sell_volume)

        if matched_volume > max_volume:
            max_volume = matched_volume
            best_price = price

    return best_price if best_price is not None else -1


def save_report_to_file(company: str, report: str, error=False):
    date = datetime.now().strftime("%Y-%m-%d")
    error_prefix = "ERROR_" if error else ""
    file_name = f"{error_prefix}{company}_{date}.md"

    file_path = os.path.join("reports", file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def collect_company_document_links(company_abbr: str):
    biznesradar_link = "https://www.biznesradar.pl/wskazniki-wartosci-rynkowej/" + company_abbr
    links = [ biznesradar_link ]
    while True:
        link = input("\nPodaj link (pusty, aby zakończyć): ").strip()
        if link == "":
            break
        links.append(link)
    return links


def collect_company_keywords():
    keywords = []
    while True:
        keyword = input("\nPodaj słowa kluczowe dla spółki do szukania newsów (pusty, aby zakończyć): ").strip()
        if keyword == "":
            break
        keywords.append(keyword)
    return keywords


if __name__ == "__main__":

    company = input("\nCzy chcesz tylko sprawdzić jedną spółkę? (nazwa spółki lub -): ").strip()
    
    if company != '-' and len(company) > 0:
        company_abbr = input("\nPodaj skrót spółki: ").strip().upper()
        sector_data = sector_2_years_data(company_abbr)
        stock_exchange = "GPW" #input("\nPodaj nazwę giełdy (GPW lub NYSE): ").strip().upper()
        if stock_exchange not in ("GPW", "NYSE"):
            print("\nNiepoprawna giełda: " + stock_exchange)
            sys.exit()
        company_document_links = collect_company_document_links(company_abbr)
        company_keywords = collect_company_keywords()
        report = analyze_company(company, company_abbr, stock_exchange, company_keywords, company_document_links, sector_data)
        save_report_to_file(company, report)
        sys.exit()

    tko = input("\nCzy chcesz tylko sprawdzić TKO? (skrót lub -): ").strip().lower()
    if tko != '-' and len(tko) > 0:
        buy_orders, sell_orders = fetch_order_books(tko)
        opening_price = calculate_opening_price(buy_orders, sell_orders)

        if opening_price != -1:
            print(f"\nOrientacyjna cena otwarcia: {opening_price:.2f} zł")
        else:
            print("\nNie udało się wyznaczyć orientacyjnej ceny otwarcia.")
        sys.exit()
