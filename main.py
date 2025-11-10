import os
import re
import time
import tempfile
import urllib.parse
from typing import List, Dict, Optional, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
import telebot

from flask import Flask
import asyncio
from aiohttp import ClientSession

# ========= НАСТРОЙКИ =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8578807833:AAGAK3_09G212WCDyxZbjbXVCE6l5YKNLnI")

# страница аккаунта по умолчанию (можно передать свою в /run <url>)
ACCOUNT_URL = "https://www.kufar.by/user/Os104G9aSmGvPEWHIflMWVI?cmp=1&cnd=2&sort=lst.d"

OUTPUT_SHEET = "Подгружаемая таблица"

# HTTP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": "https://www.kufar.by/",
}

# CSS классы
# (ваши объявления на странице аккаунта)
CARD_LEFT_CLASS = "styles_left__Y6ROt"
PRICE_CONTAINER_CLASS = "styles_price_block__Ql9um"
PRICE_CLASS = "styles_price__aVxZc"
TITLE_CLASS = "styles_title__Gx6CG"
PAGINATION_WRAP_CLASS = "styles_pagination__inner__ekQUL"
PAGE_LINK_CLASS = "styles_link__MzdxS"
ARROW_LINK_CLASS = "styles_arrow-link__O5iAx"

# (поисковая выдача конкурентов)
SELLER_CLASSES = ["styles_secondary__MzdEb", "styles_company_name__IyHuU"]
AD_BUTTON_WRAPPER_CLASS = "styles_button__wrapper__BNF9t"
CALL_BUTTON_TEXT = "Позвонить"

SCRAPE_LIMIT_PER_MODEL = 30
KEEP_TOP_N = 5

# Параллельность
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
PROGRESS_BAR_LEN = 24
EDIT_THROTTLE_SECONDS = 0.8

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# ========= УТИЛИТЫ =========
def norm_space(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def parse_price_text(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace("\u00A0", " ").replace("\u2009", " ")
    m = re.search(r"(\d[\d\s]*)", s)
    if not m:
        return None
    try:
        return float(re.sub(r"[^\d]", "", m.group(1)))
    except Exception:
        return None

def abs_url(href: str) -> str:
    if href.startswith("/"):
        return urllib.parse.urljoin("https://www.kufar.by", href)
    return href

def safe_get(url: str, timeout: float = 25.0) -> requests.Response:
    time.sleep(random.uniform(0.05, 0.2))  # лёгкий джиттер
    return requests.get(url, headers=HEADERS, timeout=timeout)

def render_bar(done: int, total: int) -> str:
    if total <= 0:
        return ""
    pct = int(done * 100 / total)
    filled = int(PROGRESS_BAR_LEN * done / total)
    return f"[{'█'*filled}{'░'*(PROGRESS_BAR_LEN-filled)}] {pct}%"

# ========= ШАГ 1. Скан аккаунта (Название, Моя цена) =========
def parse_account_list_page(html: str, page_url: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    for left in soup.find_all("div", class_=CARD_LEFT_CLASS):
        # цена
        price_val = None
        price_block = left.find("div", class_=PRICE_CONTAINER_CLASS)
        if price_block:
            p = price_block.find("p", class_=PRICE_CLASS)
            if p:
                span = p.find("span")
                if span:
                    price_val = parse_price_text(span.get_text())
            if price_val is None:
                price_val = parse_price_text(price_block.get_text(" ", strip=True))

        # название
        title_el = left.find("h3", class_=TITLE_CLASS)
        title = norm_space(title_el.get_text()) if title_el else ""

        # ссылка на товар
        link = ""
        a = left.find_parent().find("a", href=True) if left.find_parent() else None
        if a and "/item/" in a["href"]:
            link = abs_url(a["href"])
        else:
            a2 = left.find("a", href=True)
            if a2 and "/item/" in a2["href"]:
                link = abs_url(a2["href"])

        if not (title or price_val or link):
            continue

        items.append({"Название": title, "Моя цена": price_val, "Ссылка": link, "Страница": page_url})

    # пагинация
    next_url = None
    pages_seen: List[str] = []
    pwrap = soup.find("div", class_=PAGINATION_WRAP_CLASS)
    if pwrap:
        for a in pwrap.find_all("a", class_=PAGE_LINK_CLASS, href=True):
            pages_seen.append(abs_url(a["href"]))
        arrows = pwrap.find_all("a", class_=lambda c: c and ARROW_LINK_CLASS in c.split(), href=True)
        if arrows:
            next_url = abs_url(arrows[-1]["href"])
        if next_url == page_url:
            next_url = None

    return {"items": items, "next_url": next_url, "pages": pages_seen}

def discover_all_pages(start_url: str) -> List[str]:
    visited = set()
    to_visit = [start_url]
    ordered = []
    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        r = safe_get(url, timeout=25)
        r.raise_for_status()
        parsed = parse_account_list_page(r.text, url)
        ordered.append(url)
        visited.add(url)
        nxt = parsed.get("next_url")
        if nxt and nxt not in visited and nxt not in to_visit:
            to_visit.append(nxt)
        for p in parsed.get("pages", []):
            if p not in visited and p not in to_visit:
                to_visit.append(p)
    # уникализируем в исходном порядке
    seen = set()
    out = []
    for u in ordered:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def dedupe_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляем по одному экземпляру каждого товара.
    Приоритетный ключ — ссылка. Если её нет, используем (нормализованный заголовок, цена).
    """
    if df.empty:
        return df

    out = df.copy()

    # подготовим ключи
    out["link_key"] = out.get("Ссылка", "").fillna("").astype(str).str.strip()

    title_col = out.get("Модель", out.get("Название", ""))
    out["title_key"] = (
        title_col.fillna("").astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out["price_key"] = pd.to_numeric(out.get("Моя цена", None), errors="coerce").fillna(-1).astype("Int64")

    # 1) все строки со ссылкой — по одной на link_key
    with_link = out[out["link_key"] != ""].drop_duplicates(subset=["link_key"], keep="first")

    # 2) строки без ссылки — по одной на (title_key, price_key)
    no_link = out[out["link_key"] == ""].drop_duplicates(subset=["title_key", "price_key"], keep="first")

    # склеиваем и чистим служебные поля
    out = pd.concat([with_link, no_link], ignore_index=True)
    out = out.drop(columns=["link_key", "title_key", "price_key"], errors="ignore")

    return out

def scrape_account_models(url: str, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
    pages = discover_all_pages(url)
    total = len(pages)
    done = 0
    rows: List[Dict] = []
    lock = threading.Lock()
    last_edit = 0.0

    def _update():
        nonlocal last_edit
        now = time.time()
        if progress_cb and (now - last_edit >= EDIT_THROTTLE_SECONDS or done == total):
            progress_cb(done, total, "Шаг 1/2: собираю ваши товары")
            last_edit = now

    def process_page(purl: str) -> List[Dict]:
        rr = safe_get(purl, timeout=25)
        rr.raise_for_status()
        parsed = parse_account_list_page(rr.text, purl)
        return parsed["items"]

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as ex:
        futures = {ex.submit(process_page, p): p for p in pages}
        for fut in as_completed(futures):
            items = []
            try:
                items = fut.result()
            except Exception:
                items = []
            with lock:
                rows.extend(items)
                done += 1
                _update()

    # превращаем в «входной файл моделей»
    df = pd.DataFrame(rows, columns=["Название", "Моя цена", "Ссылка", "Страница"])
    df = df.rename(columns={"Название": "Модель"})
    # чистим NaN/типы
    if "Моя цена" in df.columns:
        df["Моя цена"] = pd.to_numeric(df["Моя цена"], errors="coerce")


    # >>> добавь вот это:
    before = len(df)
    df = dedupe_models(df)
    after = len(df)
    print(f"[scrape_account_models] найдено: {before}, после удаления дублей: {after}")

    return df[["Модель", "Моя цена", "Ссылка", "Страница"]]

# ========= ШАГ 2. Анализ конкурентов (как раньше) =========
def kufar_search_url(query: str) -> str:
    base = "https://www.kufar.by/l/r~minsk/deshevo"
    params = {"cmp": "1", "cnd": "2", "query": query}
    return f"{base}?{urllib.parse.urlencode(params)}"

def extract_title(card: BeautifulSoup) -> str:
    for sel in ["h3", "h2", "a", "span"]:
        el = card.select_one(sel)
        if el and norm_space(el.get_text()):
            return norm_space(el.get_text())
    return ""

def extract_price(card: BeautifulSoup) -> Optional[float]:
    block = card.find("div", class_=PRICE_CONTAINER_CLASS)
    if block:
        p = block.find("p", class_=PRICE_CLASS)
        if p:
            span = p.find("span")
            if span:
                price = parse_price_text(span.get_text())
                if price is not None:
                    return price
        pr = parse_price_text(block.get_text(" ", strip=True))
        if pr is not None:
            return pr
    return parse_price_text(card.get_text(" ", strip=True))

def extract_seller(card: BeautifulSoup) -> str:
    el = card.find("span", class_=lambda c: c and all(cls in c.split() for cls in SELLER_CLASSES))
    if el:
        return norm_space(el.get_text().replace("Продавец:", "").strip())
    cand = card.find("span", string=lambda t: t and "Продавец" in t)
    if cand:
        return norm_space(cand.get_text().replace("Продавец:", "").strip())
    return ""

def extract_url(card: BeautifulSoup) -> str:
    a = card.find("a", href=True)
    if not a:
        return ""
    href = a["href"]
    if href.startswith("/"):
        href = urllib.parse.urljoin("https://www.kufar.by", href)
    return href

def is_ad_card(card: BeautifulSoup) -> bool:
    wrapper = card.find("div", class_=AD_BUTTON_WRAPPER_CLASS)
    if wrapper and wrapper.find(string=lambda t: t and CALL_BUTTON_TEXT in t):
        return True
    if card.find(["button", "a", "span"], string=lambda t: t and CALL_BUTTON_TEXT in t):
        return True
    if card.find(string=lambda t: t and "Реклама" in t):
        return True
    return False

def select_cards(soup: BeautifulSoup) -> List[BeautifulSoup]:
    cards = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/item/" not in href:
            continue
        container = a.find_parent(["article", "section", "li", "div"]) or a
        full_href = href if not href.startswith("/") else urllib.parse.urljoin("https://www.kufar.by", href)
        if full_href in seen:
            continue
        seen.add(full_href)
        cards.append(container)
    return cards

def fetch_raw_rows(model: str) -> List[Dict]:
    url = kufar_search_url(model)
    r = safe_get(url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows: List[Dict] = []
    for card in select_cards(soup):
        if is_ad_card(card):
            continue
        title = extract_title(card)
        price = extract_price(card)
        seller = extract_seller(card)
        link = extract_url(card)
        if not (title or price or seller or link):
            continue
        rows.append({"Товар": model, "Цена, BYN": price, "Продавец": seller, "Ссылка": link, "Название": title})
        if len(rows) >= SCRAPE_LIMIT_PER_MODEL:
            break
    return rows

def remove_low_outliers(rows: List[Dict]) -> List[Dict]:
    prices = [r["Цена, BYN"] for r in rows if isinstance(r.get("Цена, BYN"), (int, float))]
    if len(prices) < 3:
        return rows
    prices_sorted = sorted(prices)
    median = pd.Series(prices_sorted).median()
    q1 = pd.Series(prices_sorted).quantile(0.25)
    q3 = pd.Series(prices_sorted).quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(q1 - 1.5 * iqr, 0.1 * median, 0)
    return [r for r in rows if (r.get("Цена, BYN") is None) or (r["Цена, BYN"] >= lower_bound)]

def format_block_for_excel(model: str, my_price: Optional[float], comp_rows: List[Dict], keep_n: int = 5) -> List[Dict]:
    rows_sorted = sorted(comp_rows, key=lambda r: 10**12 if r["Цена, BYN"] is None else r["Цена, BYN"])[:keep_n]
    comp_prices = [r["Цена, BYN"] for r in rows_sorted if isinstance(r.get("Цена, BYN"), (int, float))]
    min_price = min(comp_prices) if comp_prices else None

    diff = None
    if my_price is not None and min_price is not None:
        diff = my_price - min_price

    rec = ""
    if diff is not None:
        if diff > 0:
            rec = "снизить цену"
        elif diff < 0:
            rec = "повысить цену"
        else:
            rec = "оставить без изменений"

    out_rows: List[Dict] = []
    for idx in range(keep_n):
        r = rows_sorted[idx] if idx < len(rows_sorted) else None
        row = {
            "Товар": model if idx == 0 else "",
            "Моя цена": my_price if idx == 0 else "",
            "Цены конкурентов": (r["Цена, BYN"] if r else ""),
            "Названия конкурентов": (r["Продавец"] if r and r["Продавец"] else (r["Название"] if r else "")),
            "Минимальная цена": min_price if idx == 0 else "",
            "Разница": diff if idx == 0 else "",
            "Рекомендация": rec if idx == 0 else "",
            "Ссылка": (r["Ссылка"] if r else ""),
        }
        out_rows.append(row)
    return out_rows

def process_models_df(df_models: pd.DataFrame,
                      progress_cb: Optional[Callable[[int, int, str], None]] = None,
                      max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    # ожидаем колонки: Модель, Моя цена
    tasks = [(norm_space(str(r["Модель"])), (float(r["Моя цена"]) if pd.notna(r["Моя цена"]) else None))
             for _, r in df_models.iterrows() if norm_space(str(r["Модель"]))]

    total = len(tasks)
    done = 0
    lock = threading.Lock()
    last_edit = 0.0
    rows_all: List[Dict] = []

    def _update():
        nonlocal last_edit
        now = time.time()
        if progress_cb and (now - last_edit >= EDIT_THROTTLE_SECONDS or done == total):
            progress_cb(done, total, "Шаг 2/2: анализ рынка")
            last_edit = now

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, m, p): (m, p) for (m, p) in tasks}
        for fut in as_completed(futures):
            try:
                rows_all.extend(fut.result())
            except Exception:
                pass
            with lock:
                done += 1
                _update()

    out_df = pd.DataFrame(rows_all, columns=[
        "Товар", "Моя цена", "Цены конкурентов", "Названия конкурентов",
        "Минимальная цена", "Разница", "Рекомендация", "Ссылка"
    ])
    return out_df

def _one(model: str, my_price: Optional[float]) -> List[Dict]:
    raw = fetch_raw_rows(model)
    clean = remove_low_outliers(raw)
    return format_block_for_excel(model, my_price, clean, keep_n=KEEP_TOP_N)

def _colname(i: int) -> str:
    name = ""
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name

def autosize_columns(writer, df: pd.DataFrame, sheet_name: str):
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns, start=1):
        max_len = max([len(str(col))] + [len(str(x)) for x in df[col].tolist()])
        width = min(int(max_len * 1.2) + 2, 80)
        ws.column_dimensions[_colname(i)].width = width

def export_excel(df: pd.DataFrame, sheet: str, name_prefix: str) -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"{name_prefix}_{int(time.time())}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet, index=False)
        autosize_columns(w, df, sheet)
    return out_path

# ========= ХЕНДЛЕРЫ =========
@bot.message_handler(commands=["start", "help"])
def start_msg(message):
    bot.reply_to(message,
        "Команда:\n"
        "• /run — собрать ваши товары с аккаунта и сделать анализ цен конкурентов (Excel).\n"
        "Можно указать свой URL: /run <url_аккаунта>\n\n"
        "Покажу прогресс по двум этапам."
    )

@bot.message_handler(commands=["run"])
def run_full(message):
    # читаем URL если передан
    parts = message.text.split(maxsplit=1)
    url = ACCOUNT_URL
    if len(parts) == 2 and parts[1].startswith("http"):
        url = parts[1].strip()

    status = bot.reply_to(message, "Шаг 1/2: собираю ваши товары…\n" + render_bar(0, 1))

    try:
        # общий прогресс-коллбэк
        def progress_cb(done: int, total: int, phase: str):
            bar = render_bar(done, total)
            try:
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=status.message_id,
                    text=f"{phase}\n{bar}"
                )
            except Exception:
                pass

        # 1) Скан аккаунта -> df_models (Модель, Моя цена)
        df_models = scrape_account_models(url, progress_cb=progress_cb)
        if df_models.empty:
            bot.edit_message_text(chat_id=message.chat.id, message_id=status.message_id,
                                  text="Не нашёл товаров на странице аккаунта.")
            return

        # 2) Анализ рынка по этим моделям
        progress_cb(0, len(df_models), "Шаг 2/2: анализ рынка")
        df_out = process_models_df(df_models, progress_cb=progress_cb, max_workers=MAX_WORKERS)

        # 3) Экспорт
        xlsx_path = export_excel(df_out, sheet=OUTPUT_SHEET, name_prefix="Подгружаемая_таблица")

        try:
            bot.edit_message_text(chat_id=message.chat.id, message_id=status.message_id,
                                  text=f"Готовлю файл к отправке… ✅")
        except Exception:
            pass

        with open(xlsx_path, "rb") as f:
            bot.send_document(message.chat.id, f,
                              visible_file_name="Подгружаемая_таблица.xlsx",
                              caption="Готово ✅")

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

    try:
        bot.delete_message(chat_id=message.chat.id, message_id=status.message_id)
    except Exception:
        pass

# ========= Flask keep-alive (опционально) =========
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

def run_flask():
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "5000")))

async def keep_alive():
    url = os.getenv("KEEPALIVE_URL", "")
    if not url:
        return
    while True:
        try:
            async with ClientSession() as session:
                async with session.get(url) as resp:
                    print(f"[KeepAlive] Ping {url} → {resp.status}")
        except Exception as e:
            print(f"[KeepAlive] Ошибка пинга: {e}")
        await asyncio.sleep(300)

def start_keep_alive():
    asyncio.run(keep_alive())

# ========= ЗАПУСК =========
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=start_keep_alive, daemon=True).start()
    print("Bot is running.")
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
