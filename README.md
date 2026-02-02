# 🚨 Fraud Application Detection System - https://fraud-application-detection-vmb3m678.streamlit.app/  ( <--- Live Link )

A Machine Learning–based web application to detect **fraudulent mobile applications** by analyzing **user reviews, sentiment patterns, and rating behavior**.  
The system provides a **fraud risk score** and classifies apps as **Low, Medium, or High Risk**.

---

## 🔍 Project Overview

With the rapid growth of mobile applications, fraudulent apps have become a serious concern.  
This project aims to help users and analysts **identify suspicious applications** using review sentiment analysis and behavioral indicators.

The application processes user reviews, detects fake or suspicious content, and combines multiple signals to assess fraud risk.

---

## ✨ Key Features

- 📊 **Sentiment Analysis** on user reviews  
- 🧹 **Text Preprocessing Pipeline** (cleaning, normalization, stopword removal)  
- 🚫 **Fake / Gibberish Review Detection** using linguistic heuristics  
- ⭐ **Rating-based Risk Evaluation**  
- 📈 **Fraud Risk Score (0–100)** generation  
- ⚠️ Classification into **Low / Medium / High Risk**  
- 🌐 **Interactive Streamlit Web Interface**  
- ☁️ **Deployed on Streamlit Cloud**

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning / NLP:**  
  - scikit-learn  
  - NumPy  
  - Pandas  
- **Text Processing:** Custom NLP preprocessing logic  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub  

---



Fraud-Application-Detection/
│
├── app.py # Streamlit application
├── preprocess.py # Text preprocessing logic
├── train_model.py # Model training script
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── data/ # Review datasets
└── model/ # Saved model files (if any)

## 📂 Project Structure

1. User inputs:
   - App name  
   - User reviews (one per line)  
   - Average app rating  
2. Reviews are cleaned and analyzed for:
   - Negative sentiment  
   - Explicit scam-related keywords  
   - Fake or meaningless patterns  
3. A **Fraud Risk Score** is computed using:
   - Average sentiment score  
   - Percentage of negative reviews  
   - Number of suspicious reviews  
   - App rating  
4. The app is classified as:
   - ✅ **Low Risk (Genuine)**  
   - ⚠️ **Medium Risk (Potentially Risky)**  
   - ❌ **High Risk (Fraudulent)**  

---

## 🚀 Live Demo

🔗 **Live Application:**  
(Add your Streamlit Cloud link here)

Example:
https://fraud-application-detection-vmb3m678.streamlit.app/


---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
'bash
git clone https://github.com/mRayyanHussain/Fraud-Application-Detection.git
cd Fraud-Application-Detection
2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit app
streamlit run app.py
📊 Use Cases
Identifying fraudulent mobile applications

Analyzing fake review behavior

Academic and research projects

Resume and interview demonstrations

🎯 Future Enhancements
Integrate deep learning models for advanced sentiment analysis

Add real-time data scraping from app stores

Improve fake review detection using user behavior analytics

Add database support for historical analysis

👨‍💻 Author
M Rayyan Hussain
Computer Science & Engineering Undergraduate

🔗 GitHub: https://github.com/mRayyanHussain



