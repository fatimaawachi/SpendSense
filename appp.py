import re
import tempfile
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False


# =========================================================
# CSS
# =========================================================


st.markdown("""
<style>
/* Main app */
[data-testid="stAppViewContainer"] {
    background-color: #FAF9F6;
    color: #1E2A3A;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E2A3A;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] .stMetric {
    background-color: #24374A;
    border: 1px solid #5FB4B3;
    border-radius: 12px;
    padding: 12px;
}

/* Titles */
h1, h2, h3 {
    color: #1E2A3A;
}

h1 {
    font-weight: 800;
}

p, label, .stMarkdown {
    color: #1E2A3A;
}

/* Buttons */
.stButton > button {
    background-color: #0E7C7B;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

.stButton > button:hover {
    background-color: #5FB4B3;
    color: #1E2A3A;
    border: none;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #6B7684;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0E7C7B;
    border-bottom-color: #0E7C7B;
}

/* Tables, editable fields, and boxes */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stExpander"] {
    border: 1px solid #E4E0D6;
    border-radius: 12px;
    background-color: #FFFFFF;
}

/* Metrics and result cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E4E0D6;
    border-left: 5px solid #0E7C7B;
    border-radius: 12px;
    padding: 12px;
}

/* Information and warning messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #E4E0D6;
}
</style>
""", unsafe_allow_html=True)









# =========================================================
# SETTINGS
# =========================================================
st.set_page_config(page_title="SpendSense Bahrain", page_icon="🛒", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "easyocr_model"
HISTORY_FILE = BASE_DIR / "purchase_history.csv"

HISTORY_COLUMNS = [
    "purchase_date", "product_name", "price_bhd", "source", "receipt_reference"
]
IGNORE_WORDS = [
    "total", "subtotal", "vat", "tax", "cash", "card", "change", "balance",
    "discount", "receipt", "debit", "credit", "tender", "amount", "rounding",
]
PRICE_PATTERN = re.compile(
    r"(?:(?:BHD|B\.D\.|BD)\s*)?(\d{1,3}(?:[.,]\d{1,3}))", re.IGNORECASE
)


# =========================================================
# LOCAL CSV HISTORY
# =========================================================
def load_history():
    """Read the user's locally saved purchase records."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        data = pd.read_csv(HISTORY_FILE)
    except (OSError, pd.errors.EmptyDataError) as error:
        st.error(f"Could not open purchase_history.csv: {error}")
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in data.columns:
            data[column] = ""

    data = data[HISTORY_COLUMNS]
    data["purchase_date"] = pd.to_datetime(data["purchase_date"], errors="coerce")
    data["price_bhd"] = pd.to_numeric(data["price_bhd"], errors="coerce")
    data["product_name"] = data["product_name"].fillna("").astype(str)
    return data.dropna(subset=["purchase_date", "price_bhd"])


def write_history(data):
    """Safely save the full history to CSV using a temporary file first."""
    clean = data.copy()
    for column in HISTORY_COLUMNS:
        if column not in clean.columns:
            clean[column] = ""
    clean = clean[HISTORY_COLUMNS]

    clean["purchase_date"] = pd.to_datetime(
        clean["purchase_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    clean["price_bhd"] = pd.to_numeric(clean["price_bhd"], errors="coerce")
    clean = clean.dropna(subset=["purchase_date", "product_name", "price_bhd"])
    clean = clean[clean["product_name"].astype(str).str.strip() != ""]

    temporary_file = HISTORY_FILE.with_suffix(".temporary.csv")
    try:
        clean.to_csv(temporary_file, index=False)
        temporary_file.replace(HISTORY_FILE)
    except OSError as error:
        temporary_file.unlink(missing_ok=True)
        raise OSError(f"Could not save file at {HISTORY_FILE.resolve()}: {error}")


def save_purchases(items, source, reference):
    if items.empty:
        return 0

    old_history = load_history()
    new_rows = items[["product_name", "price_bhd"]].copy()
    new_rows["purchase_date"] = str(date.today())
    new_rows["source"] = source
    new_rows["receipt_reference"] = reference
    new_rows = new_rows[HISTORY_COLUMNS]
    write_history(pd.concat([old_history, new_rows], ignore_index=True))
    return len(new_rows)


# =========================================================
# PRETRAINED DEEP-LEARNING OCR MODEL
# =========================================================
@st.cache_resource
def load_ocr():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        ["en"],
        model_storage_directory=str(MODEL_DIR),
        download_enabled=True,
    )


# =========================================================
# OCR + RECEIPT PARSING
# =========================================================
def extract_quantity(text):
    match = re.search(r"\b(\d+)\s*(?:pcs|pc|x|pack|pk|units)\b", str(text), re.I)
    if match:
        qty = int(match.group(1))
        return max(qty, 1)
    return 1


def get_price(text):
    found = PRICE_PATTERN.findall(str(text))
    if not found:
        return None
    try:
        value = float(found[-1].replace(",", "."))
        return round(value, 3) if 0 < value <= 1000 else None
    except ValueError:
        return None


def get_product_name(text):
    text = PRICE_PATTERN.sub("", str(text))
    text = re.sub(r"\b(?:BHD|B\.D\.|BD)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9&+./ -]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -./")


def is_ignored_line(text):
    return any(word in str(text).lower() for word in IGNORE_WORDS)


def raw_ocr_table(results):
    rows = []
    for box, text, confidence in results:
        points = np.array(box)
        rows.append({
            "raw_text": str(text),
            "confidence": round(float(confidence), 2),
            "x_position": round(float(np.min(points[:, 0])), 1),
            "y_position": round(float(np.mean(points[:, 1])), 1),
        })
    return pd.DataFrame(rows)


def parse_ocr(results):
    cells = []
    for box, text, confidence in results:
        if not str(text).strip():
            continue
        points = np.array(box)
        cells.append({
            "x": float(np.min(points[:, 0])),
            "y": float(np.mean(points[:, 1])),
            "height": float(np.max(points[:, 1]) - np.min(points[:, 1])),
            "text": str(text),
            "confidence": float(confidence),
        })

    if not cells:
        return pd.DataFrame(columns=["product_name", "price_bhd", "confidence"])

    tolerance = max(12, float(np.median([cell["height"] for cell in cells])) * 0.8)
    receipt_rows = []
    for cell in sorted(cells, key=lambda item: item["y"]):
        if receipt_rows and abs(cell["y"] - receipt_rows[-1]["y"]) <= tolerance:
            receipt_rows[-1]["cells"].append(cell)
            receipt_rows[-1]["y"] = np.mean([item["y"] for item in receipt_rows[-1]["cells"]])
        else:
            receipt_rows.append({"y": cell["y"], "cells": [cell]})

    items, previous_name = [], ""
    for row in receipt_rows:
        row_cells = sorted(row["cells"], key=lambda item: item["x"])
        text = " ".join(cell["text"] for cell in row_cells)
        confidence = float(np.mean([cell["confidence"] for cell in row_cells]))
        item_price, name = get_price(text), get_product_name(text)

        if item_price is None:
            if len(name) >= 2 and not is_ignored_line(text):
                previous_name = name
            continue
        if is_ignored_line(text):
            continue
        
        qty = extract_quantity(text)
        unit_price = round(item_price / qty, 3)

        if len(name) < 2:
            name = previous_name
        if len(name) >= 2:
            items.append({
                "product_name": name,
                "price_bhd": unit_price,
                "confidence": round(confidence, 2),
            })

    return pd.DataFrame(items, columns=["product_name", "price_bhd", "confidence"]).drop_duplicates(
        subset=["product_name", "price_bhd"]
    )


def parse_voice(transcript):
    items = []
    for part in re.split(r"[,;]|\band\b", transcript, flags=re.IGNORECASE):
        item_price, name = get_price(part), get_product_name(part)
        if item_price is not None and len(name) >= 2:
            qty = extract_quantity(part)
            unit_price = round(item_price / qty, 3)
            items.append({"product_name": name, "price_bhd": unit_price, "confidence": 1.0})
    return pd.DataFrame(items, columns=["product_name", "price_bhd", "confidence"])


def read_image(image, ocr):
    results = ocr.readtext(
        np.array(image), detail=1, paragraph=False, decoder="beamsearch",
        contrast_ths=0.05, adjust_contrast=0.7, mag_ratio=2,
    )
    return parse_ocr(results), raw_ocr_table(results)


def read_video(upload, ocr):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix or ".mp4") as file:
        file.write(upload.getbuffer())
        video_path = Path(file.name)

    results = []
    try:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            return pd.DataFrame(), pd.DataFrame(), "Could not open this video."
        frame_step = max(int((video.get(cv2.CAP_PROP_FPS) or 25) * 2), 1)
        frame_number = 0
        checked_frames = 0
        while checked_frames < 60:
            success, frame = video.read()
            if not success:
                break
            if frame_number % frame_step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.extend(ocr.readtext(frame, detail=1, paragraph=False, decoder="beamsearch", contrast_ths=0.05, adjust_contrast=0.7, mag_ratio=2))
                checked_frames += 1
            frame_number += 1
        video.release()
    finally:
        video_path.unlink(missing_ok=True)
    return parse_ocr(results), raw_ocr_table(results), None


def voice_to_text(upload):
    if not HAS_SPEECH:
        return None, "For automatic voice recognition run: pip install SpeechRecognition"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as file:
        file.write(upload.getbuffer())
        audio_path = Path(file.name)
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="en-US"), None
    except sr.UnknownValueError:
        return None, "Voice was unclear. Record again or type the items manually."
    except sr.RequestError:
        return None, "Voice recognition needs internet. Type the items manually instead."
    finally:
        audio_path.unlink(missing_ok=True)


# =========================================================
# PRICE COMPARISON
# =========================================================
def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def product_similarity(one, two):
    one, two = normalise(one), normalise(two)
    if not one or not two:
        return 0
    if one in two or two in one:
        return 0.95
    string_score = SequenceMatcher(None, one, two).ratio()
    first_words, second_words = set(one.split()), set(two.split())
    word_score = len(first_words & second_words) / max(len(first_words | second_words), 1)
    return max(string_score, word_score)


def compare_price(product, current_price, saved_history):
    matches = saved_history.copy()
    matches["match_score"] = matches["product_name"].apply(
        lambda old_product: product_similarity(product, old_product)
    )
    matches = matches[matches["match_score"] >= 0.58].sort_values("purchase_date", ascending=False)

    report = {
        "last": None,
        "month": None,
        "last_message": "First saved purchase of this product.",
        "month_message": "No matching purchase from about one month ago.",
        "verdict": "Save this product now so it can be compared next time.",
        "verdict_type": "info",
    }
    if matches.empty:
        return report

    last_price = float(matches.iloc[0]["price_bhd"])
    report["last"] = last_price
    if current_price < last_price * 0.95:
        report["last_message"] = "Cheaper than your last purchase — good price."
    elif current_price > last_price * 1.05:
        report["last_message"] = "More expensive than your last purchase."
    else:
        report["last_message"] = "About the same as your last purchase."

    start = pd.Timestamp(date.today() - timedelta(days=40))
    end = pd.Timestamp(date.today() - timedelta(days=20))
    month_matches = matches[(matches["purchase_date"] >= start) & (matches["purchase_date"] <= end)]
    if not month_matches.empty:
        month_price = float(month_matches["price_bhd"].median())
        report["month"] = month_price
        if current_price < month_price * 0.95:
            report["month_message"] = "Cheaper than about one month ago — good price."
        elif current_price > month_price * 1.05:
            report["month_message"] = "Higher than about one month ago."
        else:
            report["month_message"] = "About the same as one month ago."

    baseline = min(last_price, report["month"] or last_price)
    if current_price < baseline * 0.95:
        report["verdict"] = "Great deal: lower than your saved purchase history."
        report["verdict_type"] = "success"
    elif current_price > max(baseline * 1.15, baseline + 0.100):
        report["verdict"] = "Unusual increase: over 15% above your saved price. Check item size and promotions."
        report["verdict_type"] = "warning"
    else:
        report["verdict"] = "Fair price: consistent with your saved purchase history."
    return report


# =========================================================
# STREAMLIT APP
# =========================================================
saved_history = load_history()
st.title("🛒 SpendSense Bahrain")
st.write("Scan a receipt, save its prices locally, and compare the next time you buy the same products.")
st.caption("EasyOCR is the pretrained deep-learning model. Comparisons use only your saved purchase history.")

try:
    ocr = load_ocr()
except Exception as error:
    st.error(f"EasyOCR could not start: {error}")
    st.stop()

with st.sidebar:
    st.header("Your local data")
    st.metric("Saved product prices", len(saved_history))
    st.download_button(
        "Download my purchase history",
        saved_history.to_csv(index=False).encode("utf-8"),
        "purchase_history.csv",
        "text/csv",
    )

image_tab, camera_tab, video_tab, voice_tab = st.tabs(["Image", "Camera", "Video", "Voice"])
with image_tab:
    image_upload = st.file_uploader("Upload a receipt or price-label image", type=["png", "jpg", "jpeg"])
with camera_tab:
    camera = st.camera_input("Take a clear receipt photo")
with video_tab:
    video = st.file_uploader("Upload a short receipt video", type=["mp4", "mov", "avi", "mkv"])
with voice_tab:
    audio = st.audio_input("Record product names and prices")
    manual_voice = st.text_area("Or type: Coca Cola 0.180, Lays 0.150")

if st.button("Detect products and prices", type="primary"):
    st.session_state["saved_current_scan"] = False
    with st.spinner("Detecting products and prices..."):
        if image_upload is not None:
            image = Image.open(image_upload).convert("RGB")
            items, raw_results = read_image(image, ocr)
            st.session_state.update(items=items, raw_ocr=raw_results, source="image", reference=image_upload.name)
        elif camera is not None:
            image = Image.open(camera).convert("RGB")
            items, raw_results = read_image(image, ocr)
            st.session_state.update(items=items, raw_ocr=raw_results, source="camera", reference="camera capture")
        elif video is not None:
            items, raw_results, error = read_video(video, ocr)
            if error:
                st.error(error)
            st.session_state.update(items=items, raw_ocr=raw_results, source="video", reference=video.name)
        elif audio is not None or manual_voice.strip():
            transcript = manual_voice.strip()
            if audio is not None and not transcript:
                transcript, error = voice_to_text(audio)
                if error:
                    st.warning(error)
            st.session_state.update(
                items=parse_voice(transcript) if transcript else pd.DataFrame(columns=["product_name", "price_bhd", "confidence"]),
                raw_ocr=pd.DataFrame(), source="voice", reference="voice input",
            )
        else:
            st.warning("Choose an image, camera photo, video, or voice input first.")

# ضمان ظهور قسم المراجعة والجدول دائماً حتى لو لم يتم الضغط بعد أو كانت النتائج فارغة
if "items" not in st.session_state:
    st.session_state["items"] = pd.DataFrame(columns=["product_name", "price_bhd", "confidence"])

if not st.session_state.get("raw_ocr", pd.DataFrame()).empty:
    with st.expander("Model reading details"):
        st.caption("Wrong text here means an OCR/image issue. Correct text here but wrong items below means a receipt-parsing issue.")
        st.dataframe(st.session_state["raw_ocr"], hide_index=True, use_container_width=True)

st.subheader("Review detected items before saving")
approved = st.data_editor(
    st.session_state["items"],
    column_config={
        "product_name": st.column_config.TextColumn("Product", required=True),
        "price_bhd": st.column_config.NumberColumn("Price paid (BHD)", min_value=0.001, format="%.3f"),
        "confidence": st.column_config.NumberColumn("OCR confidence", disabled=True, format="%.2f"),
    },
    num_rows="dynamic", hide_index=True, use_container_width=True, key="review_table",
)
approved["price_bhd"] = pd.to_numeric(approved["price_bhd"], errors="coerce")
approved = approved.dropna(subset=["price_bhd"])
approved = approved[approved["product_name"].astype(str).str.strip() != ""]

for _, item in approved.iterrows():
    report = compare_price(item["product_name"], float(item["price_bhd"]), saved_history)
    with st.container(border=True):
        st.markdown(f"### {item['product_name']}")
        now, last, month = st.columns(3)
        now.metric("Price now", f"{item['price_bhd']:.3f} BHD")
        last.metric("Last saved price", "No record" if report["last"] is None else f"{report['last']:.3f} BHD")
        month.metric("About one month ago", "No record" if report["month"] is None else f"{report['month']:.3f} BHD")
        st.write(f"**Last purchase:** {report['last_message']}")
        st.write(f"**One-month comparison:** {report['month_message']}")
        if report["verdict_type"] == "success":
            st.success(report["verdict"])
        elif report["verdict_type"] == "warning":
            st.warning(report["verdict"])
        else:
            st.info(report["verdict"])

if st.session_state.get("saved_current_scan"):
    st.success("This scan has already been saved. Detect a new receipt to save another purchase.")
elif st.button("Confirm and save to my purchase history"):
    try:
        saved_count = save_purchases(
            approved,
            st.session_state.get("source", "unknown"),
            st.session_state.get("reference", ""),
        )
        if saved_count:
            st.session_state["saved_current_scan"] = True
            st.success(f"Saved {saved_count} product price(s) successfully!")
            st.rerun()
        else:
            st.warning("Nothing was saved. Add a product name and price first.")
    except OSError as error:
        st.error(str(error))

st.divider()
st.subheader("Manage saved purchase history")
current_history = load_history().sort_values("purchase_date", ascending=False).reset_index(drop=True)

if current_history.empty:
    st.info("No purchases have been saved yet.")
else:
    editable_history = current_history.copy()
    editable_history.insert(0, "delete", False)
    edited_history = st.data_editor(
        editable_history,
        column_config={
            "delete": st.column_config.CheckboxColumn("Delete?", help="Tick records you want to delete permanently."),
            "purchase_date": st.column_config.DateColumn("Purchase date"),
            "product_name": st.column_config.TextColumn("Product"),
            "price_bhd": st.column_config.NumberColumn("Price (BHD)", format="%.3f"),
        },
        hide_index=True, use_container_width=True, key="history_manager",
    )
    if st.button("Delete selected saved records"):
        selected = edited_history[edited_history["delete"]]
        if selected.empty:
            st.warning("Tick at least one record to delete.")
        else:
            try:
                write_history(edited_history[~edited_history["delete"]])
                st.success(f"Deleted {len(selected)} record(s).")
                st.rerun()
            except OSError as error:
                st.error(str(error))