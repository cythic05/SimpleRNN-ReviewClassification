import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reversed = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.keras')

def decode_review(encoded_review):
    return ' '.join([reversed.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    pred = model.predict(preprocessed_input)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return sentiment, pred[0][0]

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    pred = model.predict(preprocessed_input)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {pred[0][0]:.2f}')
else:
    st.write('Please enter a movie review.')
    
