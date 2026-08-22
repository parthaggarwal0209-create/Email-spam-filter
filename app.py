import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    # Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# UI
st.set_page_config(page_title="Email Spam Classifier")

st.title("📧 Email Spam Classifier")
st.write("Enter an email or SMS message below.")

input_text = st.text_area("Message")

if st.button("Predict"):

    transformed_text = transform_text(input_text)

    vector_input = vectorizer.transform([transformed_text])

    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")