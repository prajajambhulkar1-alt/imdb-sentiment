import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vector.pkl")

# Download NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


def preprocess(text):
    text = clean_text(text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


st.title("🎬 IMDB Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    processed = preprocess(review)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")
