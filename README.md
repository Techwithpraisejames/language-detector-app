# language-detector-app
This project is a lightweight **language detection web application** built with **Streamlit** and **scikit-learn**. It uses a pre-trained machine learning model and text vectorizer to detect the language of user-provided text and report a confidence score for the prediction.

## Features

- Detects the language of any input text (17 languages for now)
- Displays prediction confidence using class probabilities
- Applies a configurable confidence threshold to avoid unreliable predictions
- Simple, interactive web interface powered by Streamlit
- Uses a saved scikit-learn model and vectorizer loaded via `joblib`

## How It Works

1. The user enters text in any language.
2. The text is transformed using a pre-trained vectorizer.
3. A trained classification model predicts the probability for each supported language.
4. The language with the highest probability is selected.
5. If the confidence score is below the threshold, the app warns the user instead of guessing.

## Try it out
The application is deployed and ready to use. Enter a short piece of text in any language and see how the model detects and scores the result in real time.
https://language-detector-app.streamlit.app/


## Tech Stack

- Python
- Streamlit
- scikit-learn
- NumPy
- joblib

## Future Improvements

- Add support for more languages and build more robust web app


