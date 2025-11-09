import os
import re
import time
import tempfile
import statistics
import urllib.parse
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
import telebot



from flask import Flask
import os
import threading
# ========= НАСТРОЙКИ =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8578807833:AAGAK3_09G212WCDyxZbjbXVCE6l5YKNLnI")
INPUT_SHEET = "Модели"           # во входном файле: колонки "Модель", "Моя цена"
COL_MODEL = "Модель"
COL_MYPRICE = "Моя цена"
OUTPUT_SHEET = "Подгружаемая таблица"

# HTTP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": "https://www.kufar.by/",
}

# CSS классы (по твоим примерам)
PRICE_CONTAINER_CLASS = "styles_price_block__Ql9um"
PRICE_CLASS = "styles_price__aVxZc"
SELLER_CLASSES = ["styles_secondary__MzdEb", "styles_company_name__IyHuU"]
AD_BUTTON_WRAPPER_CLASS = "styles_button__wrapper__BNF9t"
CALL_BUTTON_TEXT = "Позвонить"

SCRAPE_LIMIT_PER_MODEL = 30
KEEP_TOP_N = 5

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# ========= ПАРСЕР =========
def kufar_search_url(query: str) -> str:
    base = "https://www.kufar.by/l/r~minsk/deshevo"
    params = {"cmp": "1", "cnd": "2", "query": query}
    return f"{base}?{urllib.parse.urlencode(params)}"

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
                return parse_price_text(span.get_text())
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
    r = requests.get(url, headers=HEADERS, timeout=25)
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
    median = statistics.median(prices_sorted)
    q1 = statistics.median(prices_sorted[:len(prices_sorted)//2])
    q3 = statistics.median(prices_sorted[(len(prices_sorted)+1)//2:])
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

def process_model(model: str, my_price: Optional[float]) -> List[Dict]:
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

def process_excel_file(input_path: str) -> str:
    src = pd.read_excel(input_path, sheet_name=INPUT_SHEET)
    if COL_MODEL not in src.columns:
        raise ValueError(f'В листе "{INPUT_SHEET}" нет колонки "{COL_MODEL}"')
    if COL_MYPRICE not in src.columns:
        raise ValueError(f'В листе "{INPUT_SHEET}" нет колонки "{COL_MYPRICE}"')

    rows_all: List[Dict] = []
    for _, rec in src.iterrows():
        model = norm_space(str(rec[COL_MODEL]))
        my_price = rec[COL_MYPRICE]
        my_price = float(my_price) if pd.notna(my_price) else None
        rows_all.extend(process_model(model, my_price))
        time.sleep(1.0)

    out_df = pd.DataFrame(rows_all, columns=[
        "Товар", "Моя цена", "Цены конкурентов", "Названия конкурентов",
        "Минимальная цена", "Разница", "Рекомендация", "Ссылка"
    ])

    out_path = os.path.join(tempfile.gettempdir(), f"Подгружаемая_таблица_{int(time.time())}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name=OUTPUT_SHEET, index=False)
        autosize_columns(w, out_df, OUTPUT_SHEET)
    return out_path

# ========= ХЕНДЛЕРЫ =========
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message,
        "Пришлите Excel (.xlsx) с листом «Модели» и колонками:\n"
        "• Модель\n• Моя цена\n\n"
        "Я верну файл «Подгружаемая_таблица.xlsx» с 5 карточками по каждой модели."
    )

@bot.message_handler(content_types=["document"])
def handle_docs(message):
    doc = message.document
    if not doc.file_name.lower().endswith(".xlsx"):
        bot.reply_to(message, "Нужен файл .xlsx. Пришлите корректный файл.")
        return

    status = bot.reply_to(message, "Файл получен, обрабатываю…")

    try:
        # скачиваем во временный файл
        f_info = bot.get_file(doc.file_id)
        tmp_in = os.path.join(tempfile.gettempdir(), f"{int(time.time())}_{doc.file_name}")
        downloaded = bot.download_file(f_info.file_path)
        with open(tmp_in, "wb") as f:
            f.write(downloaded)

        # обработка
        tmp_out = process_excel_file(tmp_in)

        # отправка результата
        with open(tmp_out, "rb") as f:
            bot.send_document(message.chat.id, f, visible_file_name="Подгружаемая_таблица.xlsx", caption="Готово ✅")

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

    try:
        bot.delete_message(chat_id=message.chat.id, message_id=status.message_id)
    except Exception:
        pass

# Добавьте Flask app для порта
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def run_bot():
    bot.polling(none_stop=True)

# ---------------- Keep Alive ----------------
import asyncio
from aiohttp import ClientSession
import threading

async def keep_alive():
    """Периодически пингует указанный URL каждые 5 минут"""
    url = "https://kufar-uggb.onrender.com"  # замени на свой адрес

    while True:
        try:
            async with ClientSession() as session:
                async with session.get(url) as resp:
                    print(f"[KeepAlive] Ping {url} → {resp.status}")
        except Exception as e:
            print(f"[KeepAlive] Ошибка пинга: {e}")
        await asyncio.sleep(300)  # 5 минут

def start_keep_alive():
    asyncio.run(keep_alive())

# ========= ЗАПУСК =========
if __name__ == "__main__":

    # Запускаем Flask в отдельном потоке
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    #asyncio.run(keep_alive())
    
    # keep_alive — тоже в отдельном потоке
    keepalive_thread = threading.Thread(target=start_keep_alive, daemon=True)
    keepalive_thread.start()

    run_bot()

    #print("Bot is running.")
    #bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)