from numpy import e
import streamlit as st
import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.query_engine import PandasQueryEngine

import os
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = os.environ.get('OPENAI_API_KEY')

df = pd.read_csv("/content/fruit_veg_data.csv")

def get_result(df, input_text):
    
    query_engine = PandasQueryEngine(df=df, verbose=True)
    response = query_engine.query(
    input_text,
    )

    return response


def main():
    st.title("Agriculture Tool")

    # Create a text box for user input
    user_input = st.text_input("Enter your query:")

    # Create a submit button
    if st.button("Submit"):

        # Save the user's text in  variable
        user_text = user_input.lower()

        # Process the user's text with an NLP model
        processed_result = get_result(df, user_text)
        # Display the processed result
        st.success(processed_result)


if __name__ == "__main__":
    main()