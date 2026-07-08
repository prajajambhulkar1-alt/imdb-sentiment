
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Load the saved TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('vector.pkl')
model = joblib.load('model.pkl')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''Removes HTML tags, special characters, numbers, and converts to lowercase.'''
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    return text

def remove_stopwords(text):
    '''Removes common English stopwords from text.'''
    return ' '.join([word for word in str(text).split() if word not in stop_words])

def lemmatize_text(text):
    '''Applies lemmatization to words in the text.'''
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    '''Applies all preprocessing steps to raw text.'''
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def predict_sentiment(raw_text):
    '''Predicts the sentiment (positive/negative) of a given raw text.'''
    # Preprocess the raw text
    processed_text = preprocess_text(raw_text)

    # Transform the processed text using the loaded TF-IDF vectorizer
    text_vector = tfidf_vectorizer.transform([processed_text])

    # Make prediction using the loaded model
    prediction = model.predict(text_vector)

    # Decode the prediction
    # Assuming 0: negative, 1: positive based on previous LabelEncoder usage
    sentiment = 'positive' if prediction[0] == 1 else 'negative'

    return sentiment

if __name__ == "__main__":
    # Example usage
    sample_review_positive = "This movie was absolutely fantastic! I loved every moment of it. The acting was superb and the plot was engaging."
    sample_review_negative = "What a terrible film. The worst I've seen in years. Boring storyline and awful performances."
    sample_review_neutral = "The film was okay. Nothing special, but not bad either. A pretty average experience."

    print(f"Review: "{sample_review_positive}"
Predicted sentiment: {predict_sentiment(sample_review_positive)}
")
    print(f"Review: "{sample_review_negative}"
Predicted sentiment: {predict_sentiment(sample_review_negative)}
")
    print(f"Review: "{sample_review_neutral}"
Predicted sentiment: {predict_sentiment(sample_review_neutral)}
")
