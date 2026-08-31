# SpendSense Bahrain 🛒

### Personal Receipt Price Tracking Assistant

DealCheck Bahrain is a data science prototype that helps consumers track the prices they pay for everyday products over time.

The application takes receipt information through an image, camera photo, video, or voice/manual input. A pretrained **EasyOCR** deep-learning model extracts text from receipts, after which Python parsing is used to identify product names and prices. The user can review and correct the extracted information before saving it to a personal purchase history.

When the same product is purchased again, DealCheck Bahrain compares the new price with previous saved prices and identifies whether the item is **cheaper, similar, more expensive, or showing an unusual increase**.

> **Important:** The current prototype is a price-change tracking system. It does **not** claim to detect scams or prove fraud.

---

## 🎯 Problem

Consumers may not notice when the prices of everyday products increase over time.

Small changes can be difficult to remember because people purchase many different products at different times. Without a record of previous purchases, it is difficult to understand whether these changes are affecting overall spending.

DealCheck Bahrain addresses this problem by turning receipt information into a personal price history that can be used for future comparisons.

---

## 💡 Solution

The application follows four main steps:

1. **Capture**  
   The user provides receipt information through an image, camera photo, video, or voice/manual input.

2. **Extract**  
   Pretrained EasyOCR detects and extracts text from the receipt.

3. **Confirm & Save**  
   The extracted product names and prices are displayed to the user. The user can correct OCR errors before the information is saved.

4. **Compare**  
   New purchases are compared with previous prices stored in the user's purchase history.

### Overall workflow

```text
Receipt Input
     ↓
EasyOCR
     ↓
Detected Text
     ↓
Python Row Parsing
     ↓
Product Name + Price
     ↓
User Review & Correction
     ↓
purchase_history.csv
     ↓
Compare With Previous Price
     ↓
Price Change Result
```

---

## 🤖 Methodology

### Pretrained OCR Model

The main deep-learning component is **EasyOCR**, a pretrained OCR model.

OCR stands for **Optical Character Recognition**. It is used to detect text in an image and convert it into machine-readable text.

For this prototype, EasyOCR is used without custom training or fine-tuning.

The methodology is:

```text
Image / Camera / Video Frame
          ↓
     EasyOCR
          ↓
    Text Detection
          ↓
    Row Parsing
          ↓
 Product + Price Extraction
          ↓
    User Confirmation
          ↓
       Storage
```

### What Kind of a pretrained model I used?

I used EasyOCR, which is a pretrained deep-learning Optical Character Recognition (OCR) model. I used it to detect and extract text, especially product names and prices, from receipt images, camera photos, and video frames.


### why?

I used EasyOCR because receipts contain unstructured text, and I needed a way to automatically convert the information in the receipt image into machine-readable text. Using a pretrained model allowed me to use an existing deep-learning OCR system without training or fine-tuning a custom model for this prototype.


---

## 📊 Data

The project uses two main sources of data.

### 1. OCR Receipt Dataset

Receipt images are used to test the OCR and receipt-parsing workflow.

### 2. User Purchase History

Confirmed purchases become the
