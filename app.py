# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from preprocess import clean_text
# import pickle

# # -------------------------------
# # LOAD MODEL & TOKENIZER
# # -------------------------------
# model = load_model("model/lstm_model.h5")

# # IMPORTANT: Load the SAME tokenizer used during training
# with open("model/tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# MAX_LEN = 100

# # -------------------------------
# # HELPER FUNCTIONS
# # -------------------------------
# def is_gibberish(text):
#     words = text.split()
#     return len(words) < 3


# def unknown_word_ratio(text, tokenizer):
#     words = text.split()
#     if len(words) == 0:
#         return 1.0

#     known = sum(1 for w in words if w in tokenizer.word_index)
#     return 1 - (known / len(words))


# def predict_sentiment(text):
#     seq = tokenizer.texts_to_sequences([text])
#     pad = pad_sequences(seq, maxlen=MAX_LEN)
#     return model.predict(pad, verbose=0)[0][0]


# # -------------------------------
# # STREAMLIT UI
# # -------------------------------
# st.set_page_config(page_title="Fraud App Detection", layout="centered")

# st.title("📱 Fraud App Detection System")
# st.subheader("Using Sentiment Analysis & LSTM")

# st.write("Enter app reviews and rating to detect whether an application is **fraudulent or genuine**.")

# app_name = st.text_input("📌 App Name")
# reviews = st.text_area(
#     "📝 Paste App Reviews (one review per line)",
#     height=180,
#     placeholder="Example:\nThis app is very useful\nWorst app, lost my money\nFake rewards, scam"
# )
# rating = st.slider("⭐ Average App Rating", 1.0, 5.0, 3.0)

# # -------------------------------
# # PREDICTION LOGIC
# # -------------------------------
# if st.button("🔍 Check App Risk"):
#     if not reviews.strip():
#         st.warning("Please enter at least one review.")
#     else:
#         review_list = [r.strip() for r in reviews.split("\n") if r.strip()]

#         sentiment_scores = []
#         suspicious_count = 0

#         for review in review_list:
#             cleaned = clean_text(review)

#             # Suspicious review checks
#             if is_gibberish(cleaned) or unknown_word_ratio(cleaned, tokenizer) > 0.6:
#                 suspicious_count += 1
#                 continue

#             sentiment = predict_sentiment(cleaned)
#             sentiment_scores.append(sentiment)

#         # If all reviews are suspicious
#         if len(sentiment_scores) == 0:
#             avg_sentiment = 0.3  # force negative sentiment
#         else:
#             avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

#         # -------------------------------
#         # FINAL FRAUD DECISION
#         # -------------------------------
#         if (
#             avg_sentiment < 0.5
#             or suspicious_count > len(review_list) / 2
#             or rating < 3
#         ):
#             st.error("❌ HIGH RISK: Fraudulent Application Detected")
#         else:
#             st.success("✅ LOW RISK: Genuine Application")

#         # -------------------------------
#         # DISPLAY DETAILS
#         # -------------------------------
#         st.markdown("### 📊 Analysis Summary")
#         st.write("**Average Sentiment Score:**", round(avg_sentiment, 2))
#         st.write("**Total Reviews:**", len(review_list))
#         st.write("**Suspicious Reviews:**", suspicious_count)

#         if suspicious_count > 0:
#             st.warning("⚠️ Presence of fake or meaningless reviews detected")















# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from preprocess import clean_text

# def is_gibberish(text, tokenizer):
#     words = text.split()

#     # Allow short but meaningful reviews
#     if len(words) <= 2:
#         known_words = sum(1 for w in words if w in tokenizer.word_index)
#         return known_words == 0   # suspicious only if NONE are known words

#     # For longer text, check unknown ratio
#     known_words = sum(1 for w in words if w in tokenizer.word_index)
#     unknown_ratio = 1 - (known_words / len(words))

#     return unknown_ratio > 0.6



# def unknown_word_ratio(text, tokenizer):
#     words = text.split()
#     if len(words) == 0:
#         return 1.0

#     known = sum(1 for w in words if w in tokenizer.word_index)
#     return 1 - (known / len(words))

# # Load trained model and tokenizer
# model = load_model("model/lstm_model.h5")

# with open("model/tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# st.set_page_config(page_title="Fraud App Detection", layout="centered")

# st.title("📱 Fraud App Detection System")
# st.write("Detect fraudulent mobile applications using real user review analysis.")

# # User Inputs
# app_name = st.text_input("Enter App Name")

# reviews = st.text_area(
#     "Paste App Reviews (one review per line)",
#     height=150
# )

# rating = st.slider("Average App Rating", 1.0, 5.0, 3.0)

# # Prediction Button
# if st.button("Check Application"):
#     if reviews.strip() == "":
#         st.warning("Please enter at least one review.")
#     else:
#         review_list = [r.strip() for r in reviews.split("\n") if r.strip()]
#         total_reviews = len(review_list)

#         sentiment_scores = []
#         negative_reviews = 0
#         suspicious_reviews = 0

#         for review in review_list:
#             cleaned = clean_text(review)

#             # 🚫 Detect fake / meaningless reviews
#             if is_gibberish(cleaned, tokenizer):
#                 suspicious_reviews += 1
#                 continue

#             seq = tokenizer.texts_to_sequences([cleaned])
#             pad = pad_sequences(seq, maxlen=150)
#             sentiment = model.predict(pad, verbose=0)[0][0]

#             sentiment_scores.append(sentiment)

#             if sentiment < 0.5:
#                 negative_reviews += 1

#         # If all reviews are suspicious
#         if len(sentiment_scores) == 0:
#             avg_sentiment = 0.3  # force negative sentiment
#         else:
#             avg_sentiment = float(np.mean(sentiment_scores))

#         negative_percentage = (negative_reviews / total_reviews) * 100

#         # -------- Fraud Risk Score Calculation --------
#         risk_score = (
#             (1 - avg_sentiment) * 50 +
#             (negative_percentage * 0.3) +
#             max(0, (3 - rating)) * 10 +
#             (suspicious_reviews / total_reviews) * 30
#         )

#         risk_score = max(0, min(100, risk_score))
#         risk_score = round(float(risk_score), 2)

#         # Risk level
#         if risk_score >= 70:
#             risk_level = "High Risk"
#         elif risk_score >= 40:
#             risk_level = "Medium Risk"
#         else:
#             risk_level = "Low Risk"

#         # -------- Display Results --------
#         st.subheader("📊 Analysis Summary")
#         st.write("🔹 App Name:", app_name)
#         st.write("🔹 Total Reviews:", total_reviews)
#         st.write("🔹 Negative Reviews:", negative_reviews)
#         st.write("🔹 Suspicious Reviews:", suspicious_reviews)
#         st.write("🔹 Negative Review Percentage:", round(float(negative_percentage), 2), "%")
#         st.write("🔹 Average Sentiment Score:", round(float(avg_sentiment), 3))
#         st.write("🔹 Average Rating:", rating)

#         st.subheader("⚠️ Fraud Risk Assessment")
#         st.write("🔴 Fraud Risk Score:", risk_score, "%")
#         st.write("🔹 Risk Level:", risk_level)

#         # -------- Final Decision --------
#         # if risk_score >= 70:
#         #     st.error("❌ Fraudulent Application Detected")
#         # elif risk_score >= 40:
#         #     st.warning("⚠️ Potentially Risky Application")
#         # else:
#         #     st.success("✅ Genuine Application")

#         # -------- Final Decision --------
#         if suspicious_reviews > total_reviews / 2:
#             st.error("❌ Fraudulent Application Detected (Fake Review Activity)")
#         elif risk_score >= 70:
#             st.error("❌ Fraudulent Application Detected")
#         elif risk_score >= 40:
#             st.warning("⚠️ Potentially Risky Application")
#         else:
#             st.success("✅ Genuine Application")

        
#         st.write("---")
#         st.write("**Fraud Detection Logic Used:**")
#         st.write("- Sentiment analysis using LSTM")
#         st.write("- Percentage of negative reviews")
#         st.write("- Average app rating")
#         st.write("- Combined fraud risk score")













# # import streamlit as st
# # import numpy as np
# # import pickle
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from preprocess import clean_text

# # # Load trained model and tokenizer
# # model = load_model("model/lstm_model.h5")

# # with open("model/tokenizer.pkl", "rb") as f:
# #     tokenizer = pickle.load(f)

# # st.set_page_config(page_title="Fraud App Detection", layout="centered")

# # st.title("📱 Fraud App Detection System")
# # st.write("Detect fraudulent mobile applications using real user review analysis.")

# # app_name = st.text_input("Enter App Name")

# # reviews = st.text_area(
# #     "Paste App Reviews (one review per line)",
# #     height=150
# # )

# # rating = st.slider("Average App Rating", 1.0, 5.0, 3.0)

# # # if st.button("Check Application"):
# # #     if reviews.strip() == "":
# # #         st.warning("Please enter at least one review.")
# # #     else:
# # #         review_list = reviews.split("\n")
# # #         cleaned_reviews = [clean_text(r) for r in review_list]

# # #         sequences = tokenizer.texts_to_sequences(cleaned_reviews)
# # #         padded = pad_sequences(sequences, maxlen=150)

# # #         predictions = model.predict(padded)
# # #         avg_sentiment = float(np.mean(predictions))

# # #         st.subheader("Analysis Result")
# # #         st.write("🔹 Average Sentiment Score:", round(avg_sentiment, 3))
# # #         st.write("🔹 Average Rating:", rating)

# # #         # Improved fraud logic
# # #         if avg_sentiment < 0.4 and rating < 3:
# # #             st.error("❌ Fraudulent Application Detected")
# # #         else:
# # #             st.success("✅ Genuine Application")

# # #         st.write("---")
# # #         st.write("**Decision Logic:**")
# # #         st.write("- Negative sentiment + low rating → Fraud")
# # #         st.write("- Otherwise → Genuine")



# # if st.button("Check Application"):
# #     if reviews.strip() == "":
# #         st.warning("Please enter at least one review.")
# #     else:
# #         review_list = reviews.split("\n")
# #         total_reviews = len(review_list)

# #         cleaned_reviews = [clean_text(r) for r in review_list]
# #         sequences = tokenizer.texts_to_sequences(cleaned_reviews)
# #         padded = pad_sequences(sequences, maxlen=150)

# #         predictions = model.predict(padded)
# #         avg_sentiment = float(np.mean(predictions))

# #         # Review volume analysis
# #         negative_reviews = sum(pred < 0.5 for pred in predictions)
# #         negative_percentage = (negative_reviews / total_reviews) * 100

# #         st.subheader("📊 Analysis Summary")
# #         st.write("🔹 Total Reviews:", total_reviews)
# #         st.write("🔹 Negative Reviews:", negative_reviews)
# #         st.write(
# #     "🔹 Negative Review Percentage:",
# #     round(float(negative_percentage), 2),
# #     "%"
# # )
# #         st.write("🔹 Average Sentiment Score:", round(float(avg_sentiment), 3))

# #         st.write("🔹 Average Rating:", rating)

# #         # Improved fraud logic
# #         if avg_sentiment < 0.4 and rating < 3 and negative_percentage > 60:
# #             st.error("❌ Fraudulent Application Detected")
# #         else:
# #             st.success("✅ Genuine Application")

# #         st.write("---")
# #         st.write("**Fraud Detection Logic:**")
# #         st.write("- High % of negative reviews")
# #         st.write("- Low sentiment score")
# #         st.write("- Low app rating")






import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

# ----------------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------------
model = load_model("model/lstm_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150

# ----------------------------------
# NEGATIVE KEYWORDS (RULE-BASED)
# ----------------------------------
NEGATIVE_KEYWORDS = {
    "bad", "worst", "useless", "waste", "scam",
    "fraud", "fake", "poor", "terrible", "hate",
    "not useful", "not good"
}

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------
def is_explicit_negative(text):
    for word in NEGATIVE_KEYWORDS:
        if word in text:
            return True
    return False


def is_gibberish(text, tokenizer):
    words = text.split()

    # Empty text
    if len(words) == 0:
        return True

    # Short reviews (1–2 words): check if words exist in vocabulary
    if len(words) <= 2:
        known_words = sum(1 for w in words if w in tokenizer.word_index)
        return known_words == 0  # suspicious only if none are known

    # Longer text: unknown word ratio
    known_words = sum(1 for w in words if w in tokenizer.word_index)
    unknown_ratio = 1 - (known_words / len(words))

    return unknown_ratio > 0.6


def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    return model.predict(pad, verbose=0)[0][0]


# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.set_page_config(page_title="Fraud App Detection", layout="centered")

st.title("📱 Fraud App Detection System")
st.write("Detect fraudulent mobile applications using sentiment analysis and review behavior.")

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
            cleaned = clean_text(review)

            # Detect suspicious (fake/gibberish) reviews
            if is_gibberish(cleaned, tokenizer):
                suspicious_reviews += 1
                continue

            sentiment = predict_sentiment(cleaned)
            sentiment_scores.append(sentiment)

            # Negative sentiment detection
            if sentiment < 0.5 or is_explicit_negative(cleaned):
                negative_reviews += 1

        # Average sentiment
        if len(sentiment_scores) == 0:
            avg_sentiment = 0.3  # force negative if all suspicious
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

        risk_score = max(0, min(100, risk_score))
        risk_score = round(float(risk_score), 2)

        # Risk level
        if risk_score >= 70:
            risk_level = "High Risk"
        elif risk_score >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # ----------------------------------
        # DISPLAY RESULTS
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

        # ----------------------------------
        # FINAL DECISION (OVERRIDE LOGIC)
        # ----------------------------------
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
        st.write("- LSTM-based sentiment analysis")
        st.write("- Explicit negative keyword detection")
        st.write("- Fake/gibberish review identification")
        st.write("- Rating and behavior-based risk scoring")
