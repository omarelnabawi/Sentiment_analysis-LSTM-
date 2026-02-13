import sys
import os
from src import model
from src.train import model_train
from src.prediction import predict_sentiment
import streamlit as st

st.title('Sentiment Analysis')
st.write('Enter a sentence to predict what you think about the Movie:')
user_input = st.text_input('Your sentence here:')
#k = predict_sentiment(model_train(i=5),user_input)
model = model_train(i=7)
if st.button("Submit"):

    if user_input.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        
        k = predict_sentiment(model, user_input)

        if k.lower() == "positive":
            st.markdown(
                "<h2 style='color:green;'>😊 Positive</h2>",
                unsafe_allow_html=True
            )

        elif k.lower() == "negative":
            st.markdown(
                "<h2 style='color:red;'>😡 Negative</h2>",
                unsafe_allow_html=True
            )

        else:
            st.write(f"Predicted sentiment: {k}")