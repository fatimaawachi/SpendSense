# SpendSense Bahrain 🛒


#### Data link: https://www.kaggle.com/datasets/sushmithanarayan/expenses-receipt-ocr
#### App link: https://spendsense-fvstz7wr5adldkohpui3hp.streamlit.app
#### Video link: 

### Smart Purchase & Price Tracking Assistant

SpendSense Bahrain is a data science prototype designed to help consumers understand how the prices of the products they buy change over time.

The application processes purchase information from receipts using a pretrained **EasyOCR** deep-learning model. It extracts information such as product names and prices from the receipt, then organizes the extracted information into structured purchase records.

After the user reviews and confirms the extracted information, SpendSense stores the purchase history. When the user purchases the same product again, the application compares the current price with previously recorded prices and identifies whether the product has become **cheaper, similar in price, more expensive, or experienced an unusual increase**.

The main goal is to give users a clearer view of their personal spending and how product prices change over time.

---

## 🎯 Problem

Consumers often do not remember how much they previously paid for everyday products.

For example, a user might buy the same product several times:

```text
Previous purchase: 0.450 BHD
Current purchase:  0.550 BHD
```

Without a purchase history, the user may not notice that the price has increased.

SpendSense addresses this problem by extracting purchase information from receipts and building a personal history of product prices. This history can then be used to compare current purchases with previous ones.

---

## 💡 Solution

SpendSense follows a simple pipeline:

1. **Capture**  
   The user provides purchase information through a receipt image, camera input, video, or manual/voice input.

2. **Extract**  
   A pretrained EasyOCR model detects and extracts text from the receipt.

3. **Identify**  
   Python processing is used to organize the OCR output and identify product names and their corresponding prices.

4. **Confirm**  
   The user reviews the extracted information and corrects any OCR mistakes.

5. **Save**  
   Confirmed purchases are stored in the user's purchase history.

6. **Compare**  
   When the same product is purchased again, SpendSense compares its current price with previous recorded prices.

7. **Analyze**  
   The system provides a simple price-change result to help the user understand how the price has changed.

---

## 🔄 Overall Workflow

```text
Receipt / Purchase Information
             ↓
          EasyOCR
             ↓
       Extracted Text
             ↓
       Text Processing
             ↓
     Product + Price
             ↓
     User Confirmation
             ↓
      Purchase History
             ↓
     Find Previous Price
             ↓
   Compare Current Price
             ↓
       Price Change
             ↓
   Spending Insight
```

---

## 🤖 Pretrained Model

### EasyOCR

The main deep-learning component used in SpendSense is **EasyOCR**.

EasyOCR is a pretrained **Optical Character Recognition (OCR)** model.

OCR stands for **Optical Character Recognition**. It allows a computer to detect text inside an image and convert
