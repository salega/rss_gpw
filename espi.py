import random
import time
import warnings
from datetime import date, datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

warnings.simplefilter(action="ignore", category=FutureWarning)

GPW_URL = "https://www.gpw.pl/"
COMMUNICATES_URL = urljoin(GPW_URL, "komunikaty")


def _build_espi_search_url(company_keyword: str) -> str:
    return (
        f"{COMMUNICATES_URL}?"
        f"categoryRaports=EBI,ESPI"
        f"&typeRaports=RB,P,Q,O,R"
        f"&searchText={company_keyword}"
    )


def _parse_datetime_from_li(li):
    dspan = li.find("span", class_="date")
    if not dspan or not dspan.text:
        return None
    raw = dspan.get_text(" ", strip=True).replace("\xa0", " ")
    try:
        dt_str = raw[:19]
        return datetime.strptime(dt_str, "%d-%m-%Y %H:%M:%S")
    except Exception:
        return None


def _extract_title_from_li(li) -> str:
    p = li.find("p")
    return p.get_text(" ", strip=True) if p else ""


def _extract_espi_link_from_li(li):
    a = li.find("a", href=True)
    if not a:
        return None
    href = a["href"].strip()
    if not href.startswith("komunikat"):
        return None
    return urljoin(GPW_URL, href)


def _fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _find_search_results_list(soup: BeautifulSoup):
    return soup.find("ul", id="search-result")


def get_espi_links_for_company(stock_exchange: str, company_keywords=None, cutoff_days_back: int = 30):
    if stock_exchange != "GPW":
        return []

    if not company_keywords:
        return []

    cutoff = date.today() - timedelta(days=cutoff_days_back)
    seen = set()
    espi_links = []

    for company_keyword in company_keywords:
        url = _build_espi_search_url(company_keyword)
        print("Pobieranie newsów ESPI ze strony " + url)

        html = _fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        ul = _find_search_results_list(soup)
        if not ul:
            continue

        for li in ul.find_all("li"):
            dt = _parse_datetime_from_li(li)
            if not dt:
                continue
            if dt.date() < cutoff:
                continue

            full_link = _extract_espi_link_from_li(li)
            if not full_link or full_link in seen:
                continue

            title = _extract_title_from_li(li)
            espi_links.append((full_link, title, dt.strftime("%H:%M")))
            seen.add(full_link)

        time.sleep(random.uniform(0.1, 0.3))

    print("\nLinki do newsów ESPI:")
    for link, _, _ in espi_links:
        print(link)

    return sorted(espi_links)