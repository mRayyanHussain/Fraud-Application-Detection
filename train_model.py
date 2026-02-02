import pandas as pd
from preprocess import clean_text

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
data = pd.read_csv("data/google_play_reviews.csv")

# Keep only required columns
data = data[['content', 'score']]

# Rename columns for clarity
data.columns = ['review', 'rating']

# Remove missing reviews
data.dropna(subset=['review'], inplace=True)

# Create sentiment from rating
def rating_to_sentiment(rating):
    if rating >= 4:
        return 1   # positive
    elif rating <= 2:
        return 0   # negative
    else:
        return None  # neutral

data['sentiment'] = data['rating'].apply(rating_to_sentiment)

# Drop neutral reviews
data.dropna(subset=['sentiment'], inplace=True)

# Clean review text
data['cleaned_review'] = data['review'].astype(str).apply(clean_text)

X = data['cleaned_review']
y = data['sentiment']

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, maxlen=150)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42
)

# Improved LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training started on real dataset...")
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save model & tokenizer
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model trained on real dataset and saved successfully!")
