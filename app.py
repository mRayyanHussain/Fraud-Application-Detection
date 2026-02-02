import streamlit as st
import numpy as np
import pickle
import re
from preprocess import clean_text

# ----------------------------------
# STREAMLIT CONFIG
# ----------------------------------
st.set_page_config(page_title="Fraud App Detection", layout="centered")

# ----------------------------------
# SIMPLE KEYWORD-BASED SENTIMENT
# (Streamlit-safe replacement for LSTM)
# ----------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "nice",
    "useful", "amazing", "love", "best", "perfect"
}

NEGATIVE_WORDS = {
    "bad", "worst", "useless", "waste", "scam",
    "fraud", "fake", "poor", "terrible", "hate",
    "not useful", "not good", "money lost"
}

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------
def is_explicit_negative(text):
    for word in NEGATIVE_WORDS:
        if word in text:
            return True
    return False


def sentiment_score(text):
    """Returns score between 0 and 1"""
    words = text.split()
    if len(words) == 0:
        return 0.5

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)

    score = (pos + 1) / (pos + neg + 2)
    return score


def is_gibberish(text):
    words = text.split()
    if len(words) <= 2:
        return True
    return not any(w.isalpha() for w in words)


# ----------------------------------
# UI
# ----------------------------------
st.title("📱 Fraud App Detection System")
st.write("Detect fraudulent mobile applications using review behavior and sentiment analysis.")

app_name = st.text_input("🔹 Enter App Name")

reviews = st.text_area(
    "📝 Paste App Reviews (one review per line)",
    height=180,
    placeholder="Example:\nbad\nnot useful\nThis app is a scam\nGreat app"
)

rating = st.slider("⭐ Average App Rating", 1.0, 5.0, 3.0)

# ----------------------------------
# MAIN LOGIC
# ----------------------------------
if st.button("🔍 Check Application"):
    if not reviews.strip():
        st.warning("Please enter at least one review.")
    else:
        review_list = [r.strip() for r in reviews.split("\n") if r.strip()]
        total_reviews = len(review_list)

        sentiment_scores = []
        negative_reviews = 0
        suspicious_reviews = 0

        for review in review_list:
            cleaned = clean_text(review.lower())

            if is_gibberish(cleaned):
                suspicious_reviews += 1
                continue

            score = sentiment_score(cleaned)
            sentiment_scores.append(score)

            if score < 0.45 or is_explicit_negative(cleaned):
                negative_reviews += 1

        if len(sentiment_scores) == 0:
            avg_sentiment = 0.3
        else:
            avg_sentiment = float(np.mean(sentiment_scores))

        negative_percentage = (negative_reviews / total_reviews) * 100

        # ----------------------------------
        # FRAUD RISK SCORE
        # ----------------------------------
        risk_score = (
            (1 - avg_sentiment) * 50 +
            (negative_percentage * 0.3) +
            max(0, (3 - rating)) * 10 +
            (suspicious_reviews / total_reviews) * 30
        )

        risk_score = round(max(0, min(100, risk_score)), 2)

        if risk_score >= 70:
            risk_level = "High Risk"
        elif risk_score >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # ----------------------------------
        # RESULTS
        # ----------------------------------
        st.subheader("📊 Analysis Summary")
        st.write("🔹 App Name:", app_name)
        st.write("🔹 Total Reviews:", total_reviews)
        st.write("🔹 Negative Reviews:", negative_reviews)
        st.write("🔹 Suspicious Reviews:", suspicious_reviews)
        st.write("🔹 Negative Review Percentage:", round(negative_percentage, 2), "%")
        st.write("🔹 Average Sentiment Score:", round(avg_sentiment, 3))
        st.write("🔹 Average Rating:", rating)

        st.subheader("⚠️ Fraud Risk Assessment")
        st.write("🔴 Fraud Risk Score:", risk_score, "%")
        st.write("🔹 Risk Level:", risk_level)

        if suspicious_reviews > total_reviews / 2:
            st.error("❌ Fraudulent Application Detected (Fake Review Activity)")
        elif risk_score >= 70:
            st.error("❌ Fraudulent Application Detected")
        elif risk_score >= 40:
            st.warning("⚠️ Potentially Risky Application")
        else:
            st.success("✅ Genuine Application")

        st.write("---")
        st.write("**Fraud Detection Logic Used:**")
        st.write("- Sentiment analysis (lightweight, deploy-safe)")
        st.write("- Negative keyword detection")
        st.write("- Fake/gibberish review identification")
        st.write("- Rating and behavior-based risk scoring")
