import streamlit as st
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

with open("LogisticRegression.pkl", "rb") as f:
    LR = pickle.load(f)
with open("DecisionTree.pkl", "rb") as f:
    DT = pickle.load(f)
with open("GradientBoosting.pkl", "rb") as f:
    GB = pickle.load(f)
with open("RandomForest.pkl", "rb") as f:
    RF = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorization = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
st.title("📰 Fake News Detection using Machine Learning")
st.write("Enter a news article below to check if it's Fake or Real.")

news_input = st.text_area("Enter News Text:")
if st.button("Detect Fake News"):
    if news_input:
        cleaned_text = clean_text(news_input)
        transformed_text = vectorization.transform([cleaned_text])

        pred_LR = LR.predict(transformed_text)[0]
        pred_DT = DT.predict(transformed_text)[0]
        pred_GB = GB.predict(transformed_text)[0]
        pred_RF = RF.predict(transformed_text)[0]
        
        def output_label(n):
            return "Fake News" if n == 0 else "Real News"
        st.subheader("Predictions:")
        st.write(f"**Logistic Regression:** {output_label(pred_LR)}")
        st.write(f"**Decision Tree:** {output_label(pred_DT)}")
        st.write(f"**Gradient Boosting:** {output_label(pred_GB)}")
        st.write(f"**Random Forest:** {output_label(pred_RF)}")
    else:
        st.warning("Please enter some news text.")
