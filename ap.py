import re
from datetime import date, timedelta
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import speech_recognition as sr
except ImportError:
    sr = None


st.set_page_config(page_title="SpendSense Bahrain",
                    page_icon="🛒",
                    layout="wide")





# The app's style

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


/* Tables and boxes */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stExpander"] {
    border: 1px solid #E4E0D6;
    border-radius: 12px;
    background-color: #FFFFFF;
}


/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E4E0D6;
    border-left: 5px solid #0E7C7B;
    border-radius: 12px;
    padding: 12px;
}


/* Messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}


/* Divider */
hr {
    border-color: #E4E0D6;
}

</style>
""", unsafe_allow_html=True)








folder = Path(__file__).parent
model_folder = folder / "easyocr_model"
history_file = folder / "purchase_history.csv"

columns = ["purchase_date",
            "product_name",
            "price_bhd",
            "source",
            "receipt_reference"]

ignore_words = ["total",
                "subtotal",
                "vat",
                "tax",
                "cash",
                "card",
                "change",
                "balance",
                "discount",
                "receipt",
                "debit",
                "credit",
                "tender",
                "amount",
                "rounding"]







def load_history():
    if not history_file.exists():
        return pd.DataFrame(columns=columns)

    try:
        data = pd.read_csv(history_file)
    except:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in data.columns:
            data[col] = ""

    data = data[columns]

    data["purchase_date"] = pd.to_datetime(data["purchase_date"], errors="coerce")

    data["price_bhd"] = pd.to_numeric(data["price_bhd"], errors="coerce")

    data["product_name"] = data["product_name"].fillna("")

    data = data.dropna(subset=["purchase_date", "price_bhd"])

    return data









def save_history(data):
    data = data.copy()

    for col in columns:
        if col not in data.columns:
            data[col] = ""

    data = data[columns]

    data["purchase_date"] = pd.to_datetime(data["purchase_date"], errors="coerce")

    data["price_bhd"] = pd.to_numeric(data["price_bhd"], errors="coerce")

    data = data.dropna(subset=["purchase_date", "product_name", "price_bhd"])

    data = data[data["product_name"].astype(str).str.strip() != ""]

    data["purchase_date"] = data["purchase_date"].astype(str)

    data.to_csv(history_file, index=False)












def add_to_history(items, source, reference):
    if items.empty:
        return 0

    old = load_history()

    new = items[["product_name", "price_bhd"]]

    new["purchase_date"] = str(date.today())
    new["source"] = source
    new["receipt_reference"] = reference

    new = new[columns]

    save_history(pd.concat([old, new], ignore_index=True))

    return len(new)











@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], model_storage_directory=str(model_folder), download_enabled=True)







def get_price(text):
    words = str(text).split()

    for word in reversed(words):
        if "." in word or "," in word:
            try:
                price = float(word.replace(",", "."))

                if price > 0:
                    return price

            except:
                pass

    return None










def get_name(text):

    text = str(text)

    if ":" in text:
        text = text.split(":")[0]

    text = text.replace("IPC", "")
    text = text.replace("1PC", "")
    text = text.replace("2PCS", "")
    text = text.replace("PCS", "")
    text = text.replace("PC", "")
    text = text.replace("Rece", "")

    return text.strip()









def get_quantity(text):
    words = str(text).split()

    for word in words:
        if word.isdigit():
            number = int(word)

            if number > 0:
                return number

        if word.upper().endswith("PCS"):
            number = word[:-3]

            if number.isdigit():
                return int(number)

    return 1












def ignored(text):
    words = str(text).lower().split()

    for word in ignore_words:
        if word in words:
            return True

    return False













def raw_results_table(results):
    data = []

    for box, text, confidence in results:
        points = np.array(box)

        data.append({"text": str(text),
                    "confidence": round(float(confidence), 2),
                    "x": round(float(np.min(points[:, 0])), 1),
                    "y": round(float(np.mean(points[:, 1])), 1)})

    return pd.DataFrame(data)














def parse_receipt(results):

    detected = []

    for box, text, confidence in results:

        if not str(text).strip():
            continue

        points = np.array(box)

        detected.append({"x": float(np.min(points[:, 0])),
                        "y": float(np.mean(points[:, 1])),
                        "height": float(np.max(points[:, 1]) - np.min(points[:, 1])),
                        "text": str(text),
                        "confidence": float(confidence)})

    if not detected:
        return pd.DataFrame(columns=["product_name",
                                    "price_bhd",
                                    "confidence"])

    heights = []

    for item in detected:
        heights.append(item["height"])

    tolerance = max(12, np.median(heights) * 0.8)

    rows = []

    while len(detected) > 0:
        first = detected[0]
        same_row = []

        for item in detected:

            if abs(item["y"] - first["y"]) <= tolerance:
                same_row.append(item)

        for item in same_row:
            detected.remove(item)

        rows.append({"y": first["y"], "items": same_row})

    products = []
    last_name = ""

    for row in rows:
        row_items = row["items"]
        changed = True

        while changed:
            changed = False

            for i in range(len(row_items) - 1):
                if row_items[i]["x"] > row_items[i + 1]["x"]:
                    temp = row_items[i]
                    row_items[i] = row_items[i + 1]
                    row_items[i + 1] = temp
                    changed = True

        text = ""

        for item in row_items:
            text = text + " " + item["text"]

        text = text.strip()

        confidence = 0

        for item in row_items:
            confidence = confidence + item["confidence"]

        confidence = confidence / len(row_items)

        price = get_price(text)
        name = get_name(text)

        if price is None:

            if len(name) >= 2 and not ignored(text):
                last_name = name

            continue

        if ignored(text):
            continue

        quantity = get_quantity(text)
        price = round(price / quantity, 3)

        if len(name) < 2:
            name = last_name

        if len(name) >= 2:
            products.append({"product_name": name,
                            "price_bhd": price,
                            "confidence": round(float(confidence), 2)})

    return pd.DataFrame(products,columns=["product_name",
                                        "price_bhd",
                                        "confidence"]).drop_duplicates(subset=["product_name",
                                                                                "price_bhd"])









def read_image(image, ocr):
    results = ocr.readtext(np.array(image), detail=1, paragraph=False, decoder="beamsearch", contrast_ths=0.05, adjust_contrast=0.7, mag_ratio=2)

    return (parse_receipt(results), raw_results_table(results))



















def parse_voice(text):

    parts = text.split(" and ")

    items = []

    for part in parts:

        words = part.strip().split()

        if len(words) < 2:
            continue

        try:
            price = float(words[-1])
            name = " ".join(words[:-1])

            items.append({"product_name": name,
                        "price_bhd": price,
                        "confidence": 1.0})

        except:
            continue

    return pd.DataFrame(items, columns=["product_name",
                                        "price_bhd",
                                        "confidence"])











def voice_to_text(audio):

    if audio is None:
        return ""

    try:
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio) as source:
            recording = recognizer.record(source)

        text = recognizer.recognize_google(recording, language="en-US")

        return text

    except:
        return ""

















def similarity(a, b):

    a = str(a).lower()
    b = str(b).lower()

    if a == b:
        return 1

    words = a.split()
    count = 0

    for word in words:
        if word in b:
            count = count + 1

    if len(words) == 0:
        return 0

    return count / len(words)























def check_price(product, price, history):

    if history.empty:
        return {"last": None,
                "month": None,
                "last_text": "No previous purchase.",
                "month_text": "No purchase from about one month ago.",
                "message": "Save this price to compare it next time.",
                "type": "info"}

    scores = []

    for name in history["product_name"]:
        score = similarity(product, name)
        scores.append(score)

    history["score"] = scores
    matches = history[history["score"] >= 0.58]
    matches = matches.sort_values("purchase_date", ascending=False)

    if matches.empty:
        return {"last": None,
                "month": None,
                "last_text": "No previous purchase.",
                "month_text": "No purchase from about one month ago.",
                "message": "Save this price to compare it next time.",
                "type": "info"}

    last_price = float(matches.iloc[0]["price_bhd"])

    if price < last_price * 0.95:
        last_text = "Cheaper than your last purchase."

    elif price > last_price * 1.05:
        last_text = "More expensive than your last purchase."

    else:
        last_text = "About the same as your last purchase."

    start = pd.Timestamp(date.today() - timedelta(days=40))
    end = pd.Timestamp(date.today() - timedelta(days=20))
    old = matches[(matches["purchase_date"] >= start) & (matches["purchase_date"] <= end)]
    month_price = None

    if not old.empty:
        month_price = float(old["price_bhd"].median())

        if price < month_price * 0.95:
            month_text = "Cheaper than about one month ago."

        elif price > month_price * 1.05:
            month_text = "Higher than about one month ago."

        else:
            month_text = "About the same as one month ago."

    else:
        month_text = "No purchase from about one month ago."

    if month_price is None:
        lowest = last_price
    else:
        lowest = min(last_price, month_price)

    if price < lowest * 0.95:
        message = "Good deal. This price is lower than your saved prices."
        message_type = "success"

    elif price > lowest * 1.15:
        message = "Price is higher than usual. Check the size or promotion."
        message_type = "warning"

    else:
        message = "Fair price compared with your saved prices."
        message_type = "info"

    return {"last": last_price,
            "month": month_price,
            "last_text": last_text,
            "month_text": month_text,
            "message": message,
            "type": message_type}


history = load_history()











st.title("🛒 SpendSense Bahrain")
st.write("Scan your receipt and keep track of the prices you pay.")
st.caption("EasyOCR reads the receipt and the saved history is used for price comparison.")


try:
    ocr = load_ocr()
except Exception as e:
    st.error(f"EasyOCR could not start: {e}")
    st.stop()


with st.sidebar:
    st.header("My data")
    st.metric("Saved prices", len(history))

    st.download_button("Download history",
                        history.to_csv(index=False).encode("utf-8"),
                        "purchase_history.csv",
                        "text/csv")


tab1, tab2, tab3 = st.tabs(["Image", "Camera", "Voice"])

with tab1:
    image_upload = st.file_uploader("Upload a receipt", type=["png", "jpg", "jpeg"])

with tab2:
    camera = st.camera_input("Take a receipt photo")


with tab3:
    audio = st.audio_input("Record the products and prices")

    typed_voice = st.text_area("Or type them", placeholder="Coca Cola 0.180, Lays 0.150")


if st.button("Detect products and prices", type="primary"):
    st.session_state["saved"] = False

    if image_upload:
        image = Image.open(image_upload).convert("RGB")
        items, raw = read_image(image, ocr)

        st.session_state["items"] = items
        st.session_state["raw"] = raw
        st.session_state["source"] = "image"
        st.session_state["reference"] = image_upload.name

    elif camera:
        image = Image.open(camera).convert("RGB")
        items, raw = read_image(image, ocr)

        st.session_state["items"] = items
        st.session_state["raw"] = raw
        st.session_state["source"] = "camera"
        st.session_state["reference"] = "camera"

    
    elif audio or typed_voice.strip():
        text = typed_voice.strip()

        if audio and not text:
            text = voice_to_text(audio)

        if text:
            items = parse_voice(text)
        else:
            items = pd.DataFrame(columns=["product_name",
                                        "price_bhd",
                                        "confidence"])

        st.session_state["items"] = items
        st.session_state["raw"] = pd.DataFrame()
        st.session_state["source"] = "voice"
        st.session_state["reference"] = "voice input"

    else:
        st.warning("Choose an input first.")


if "items" not in st.session_state:

    st.session_state["items"] = pd.DataFrame(columns=["product_name",
                                                    "price_bhd",
                                                    "confidence"])


if not st.session_state.get("raw", pd.DataFrame()).empty:

    with st.expander("OCR result"):
        st.dataframe(st.session_state["raw"],hide_index=True, use_container_width=True)


st.subheader("Check the detected items")
items = st.data_editor(st.session_state["items"], column_config={"product_name": st.column_config.TextColumn("Product"),
                                                                 "price_bhd": st.column_config.NumberColumn("Price (BHD)",),
                                                                 "confidence": st.column_config.NumberColumn("Confidence",disabled=True,)
                                                                 },num_rows="dynamic",
                                                                 hide_index=True,
                                                                 use_container_width=True,
                                                                 key="items_table")


items["price_bhd"] = pd.to_numeric(items["price_bhd"], errors="coerce")
items = items.dropna(subset=["price_bhd"])
items = items[items["product_name"].astype(str).str.strip() != ""]






for _, item in items.iterrows():
    result = check_price(item["product_name"], float(item["price_bhd"]), history)

    with st.container(border=True):
        st.markdown(f"### {item['product_name']}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Price now", f"{item['price_bhd']} BHD")

        if result["last"] is None:
            col2.metric("Last price", "No record")

        else:
            col2.metric("Last price", f"{result['last']} BHD")

        if result["month"] is None:
            col3.metric("About one month ago", "No record")

        else:
            col3.metric("About one month ago", f"{result['month']} BHD")

        st.write("Last purchase:", result["last_text"])
        st.write("One-month comparison:", result["month_text"])

        if result["type"] == "success":
            st.success(result["message"])

        elif result["type"] == "warning":
            st.warning(result["message"])

        else:
            st.info(result["message"])


if st.session_state.get("saved"):
    st.success("This scan was already saved.")

elif st.button("Save these prices"):
    count = add_to_history(items, st.session_state.get("source", "unknown"),
                                  st.session_state.get("reference", ""))

    if count > 0:
        st.session_state["saved"] = True
        st.success(f"Saved {count} product(s).")
        st.rerun()

    else:
        st.warning("There is nothing to save.")












st.divider()
st.subheader("Purchase history")

history = load_history()

if history.empty:
    st.info("No saved purchases yet.")

else:
    history = history.sort_values("purchase_date", ascending=False).reset_index(drop=True)

    history.insert(0, "delete", False)

    edited = st.data_editor(
        history,
        column_config={"delete": st.column_config.CheckboxColumn("Delete"),
                        "purchase_date": st.column_config.DateColumn("Date"),
                        "product_name": st.column_config.TextColumn("Product"),
                        "price_bhd": st.column_config.NumberColumn("Price (BHD)")},hide_index=True,
                                                                                    use_container_width=True,
                                                                                    key="history_table")

    if st.button("Delete selected"):
        selected = edited[edited["delete"] == True]

        if selected.empty:
            st.warning("Select something to delete.")

        else:
            save_history(edited[edited["delete"] == False])
            st.success(f"Deleted {len(selected)} record(s).")
            st.rerun()
