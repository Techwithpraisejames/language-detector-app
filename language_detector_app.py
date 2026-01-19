import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
language_model = joblib.load(r'C:\Users\HP-PC\Downloads\language_model.joblib', mmap_mode='r')
vectorizer = joblib.load(r'C:\Users\HP-PC\Downloads\vectorizer.joblib', mmap_mode='r')

st.title('Language Detection Application')

user_input = st.text_input('Enter a text in any language:')

confidence_threshold = 0.8  # You can adjust this threshold

if user_input:
    # Transform the user input using the loaded vectorizer
    data = vectorizer.transform([user_input]).toarray()

    # Get the probability distribution for each class
    probabilities = language_model.predict_proba(data)[0]

    # Get the index of the class with the highest probability
    predicted_class_idx = np.argmax(probabilities)

    # Get the predicted language and its probability
    predicted_language = language_model.classes_[predicted_class_idx]
    max_probability = probabilities[predicted_class_idx]

    if max_probability >= confidence_threshold:
        st.write(f"Detected Language: {predicted_language} (Confidence: {max_probability:.2f})")
    else:
        st.write(f"Could not confidently detect the language. Please try again with a different text. (Highest confidence for {predicted_language}: {max_probability:.2f})")
