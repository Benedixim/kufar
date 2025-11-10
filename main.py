import os
import re
import time
import tempfile
import urllib.parse
from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
import telebot

from flask import Flask
import asyncio
from aiohttp import ClientSession
import random

# ========= НАСТРОЙКИ =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8578807833:AAGAK3_09G212WCDyxZbjbXVCE6l5YKNLnI")

# Базовый URL аккаунта (можно переопределить аргументом /scan <url>)
ACCOUNT_URL = "https://www.kufar.by/user/Os104G9aSmGvPEWHIflMWVI?cmp=1&cnd=2&sort=lst.d"

# HTTP заголовки
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": "https://www.kufar.by/",
}

# CSS-классы (из твоего примера)
CARD_LEFT_CLASS = "styles_left__Y6ROt"
PRICE_CONTAINER_CLASS = "styles_price_block__Ql9um"
PRICE_CLASS = "styles_price__aVxZc"
TITLE_CLASS = "styles_title__Gx6CG"

# Пагинация
PAGINATION_WRAP_CLASS = "styles_pagination__inner__ekQUL"  # data-cy="account-listing-pagination"
PAGE_LINK_CLASS = "styles_link__MzdxS"
ARROW_LINK_CLASS = "styles_arrow-link__O5iAx"  # у "стрелок" (< и >)

# Параллелизм для ускорения (по страницам)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
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
    # чуть-чуть джиттера, чтобы не долбить синхронно
    time.sleep(random.uniform(0.05, 0.2))
    return requests.get(url, headers=HEADERS, timeout=timeout)

def render_bar(done: int, total: int) -> str:
    if total <= 0:
        return ""
    pct = int(done * 100 / total)
    filled = int(PROGRESS_BAR_LEN * done / total)
    return f"[{'█'*filled}{'░'*(PROGRESS_BAR_LEN-filled)}] {pct}%"

# ========= ПАРСИНГ СТРАНИЦЫ СПИСКА =========
def parse_list_page(html: str, page_url: str) -> Dict:
    """
    Возвращает:
      items: List[Dict{Название, Цена, Ссылка}]
      next_url: str | None
      pages: List[str] (все видимые ссылки-страницы; нужно для общего плана)
    """
    soup = BeautifulSoup(html, "html.parser")

    # Карточки
    items: List[Dict] = []
    for left in soup.find_all("div", class_=CARD_LEFT_CLASS):
        # Цена
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

        # Название
        title_el = left.find("h3", class_=TITLE_CLASS)
        title = norm_space(title_el.get_text()) if title_el else ""

        # Ссылка (ищем ближайшую родительскую <a href="/item/...">)
        link = ""
        a = left.find_parent().find("a", href=True) if left.find_parent() else None
        if a and "/item/" in a["href"]:
            link = abs_url(a["href"])
        else:
            # запасной поиск внутри соседних контейнеров
            a2 = left.find("a", href=True)
            if a2 and "/item/" in a2["href"]:
                link = abs_url(a2["href"])

        if not (title or price_val or link):
            continue

        items.append({
            "Название": title,
            "Цена, BYN": price_val,
            "Ссылка": link
        })

    # Пагинация
    next_url = None
    pages_seen: List[str] = []
    pwrap = soup.find("div", class_=PAGINATION_WRAP_CLASS)
    if pwrap:
        links = pwrap.find_all("a", class_=PAGE_LINK_CLASS, href=True)
        for a in links:
            href = abs_url(a["href"])
            pages_seen.append(href)
        # эвристика "next": ссылка-стрелка справа (имеет класс styles_arrow-link__O5iAx и href)
        right_arrows = pwrap.find_all("a", class_=lambda c: c and ARROW_LINK_CLASS in c.split(), href=True)
        if right_arrows:
            # берём последнюю стрелку (вправо)
            next_url = abs_url(right_arrows[-1]["href"])

        # иногда "стрелка" есть, но она ведёт на текущую/предыдущую — на всякий случай фильтр:
        if next_url and next_url == page_url:
            next_url = None

    return {"items": items, "next_url": next_url, "pages": pages_seen}

def discover_all_pages(start_url: str) -> List[str]:
    """
    Сначала грузим первую страницу, собираем видимые номера,
    потом идём по "стрелке" next, пока есть.
    """
    visited = set()
    to_visit = [start_url]
    all_pages = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        r = safe_get(url, timeout=25)
        r.raise_for_status()
        parsed = parse_list_page(r.text, url)
        all_pages.append(url)
        visited.add(url)

        # если есть next — добавим в очередь
        nxt = parsed.get("next_url")
        if nxt and nxt not in visited and nxt not in to_visit:
            to_visit.append(nxt)

        # также смотрим явные номера страниц, чтобы не пропустить
        for p in parsed.get("pages", []):
            if p not in visited and p not in to_visit:
                to_visit.append(p)

    # уникализируем с сохранением порядка
    seen = set()
    ordered = []
    for u in all_pages:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

def scrape_account(url: str,
                   progress_cb: Optional[callable] = None) -> pd.DataFrame:
    """
    Обходит все страницы аккаунта и собирает товары: Название, Цена, Ссылка, Страница.
    Страницы ходим параллельно для ускорения.
    """
    pages = discover_all_pages(url)
    total = len(pages)
    done = 0
    lock = threading.Lock()
    last_edit = 0.0

    rows: List[Dict] = []

    def _update():
        nonlocal last_edit
        now = time.time()
        if progress_cb and (now - last_edit >= EDIT_THROTTLE_SECONDS or done == total):
            progress_cb(done, total)
            last_edit = now

    def process_page(purl: str) -> List[Dict]:
        r = safe_get(purl, timeout=25)
        r.raise_for_status()
        parsed = parse_list_page(r.text, purl)
        out = []
        for it in parsed["items"]:
            out.append({
                "Название": it["Название"],
                "Цена, BYN": it["Цена, BYN"],
                "Ссылка": it["Ссылка"],
                "Страница": purl
            })
        return out

    # Параллельно по страницам
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_page, p): p for p in pages}
        for fut in as_completed(futures):
            page_rows = []
            try:
                page_rows = fut.result()
            except Exception:
                page_rows = []
            with lock:
                rows.extend(page_rows)
                done += 1
                _update()

    df = pd.DataFrame(rows, columns=["Название", "Цена, BYN", "Ссылка", "Страница"])
    # можно отсортировать, например, по убыванию даты (страницы уже в порядке переходов),
    # но надёжнее — по цене возрастанию:
    df = df.sort_values(by=["Цена, BYN", "Название"], na_position="last").reset_index(drop=True)
    return df

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

def export_excel(df: pd.DataFrame) -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"Товары_{int(time.time())}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Товары", index=False)
        autosize_columns(w, df, "Товары")
    return out_path

# ========= ХЕНДЛЕРЫ =========
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message,
        "Команда:\n"
        "• /scan — собрать все товары со страницы аккаунта и вернуть Excel.\n\n"
        "Можно указать свой URL:\n"
        "/scan https://www.kufar.by/user/XXXXX?cmp=1&cnd=2&sort=lst.d\n"
        "Пока считаю — покажу прогресс по страницам."
    )

@bot.message_handler(commands=["scan"])
def on_scan(message):
    # парсим аргумент URL, если передан
    parts = message.text.split(maxsplit=1)
    url = ACCOUNT_URL
    if len(parts) == 2 and parts[1].startswith("http"):
        url = parts[1].strip()

    status = bot.reply_to(message, "Начинаю сканирование…\n[░░░░░░░░░░░░░░░░░░░░░░] 0%")

    try:
        def progress_cb(done: int, total: int):
            bar = render_bar(done, total)
            try:
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=status.message_id,
                    text=f"Обработка страниц: {done}/{total}\n{bar}"
                )
            except Exception:
                pass

        df = scrape_account(url, progress_cb=progress_cb)
        xlsx_path = export_excel(df)

        try:
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status.message_id,
                text=f"Найдено позиций: {len(df)}. Готовлю файл… ✅"
            )
        except Exception:
            pass

        with open(xlsx_path, "rb") as f:
            bot.send_document(
                message.chat.id, f,
                visible_file_name="Товары.xlsx",
                caption="Готово ✅"
            )

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

    # пробуем удалить статус
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
    # фоновые сервисы (если нужны)
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=start_keep_alive, daemon=True).start()

    print("Bot is running.")
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
