# import streamlit as st
# import pandas as pd
# import joblib
# from utils import preprocessor

# def run():
#     model = joblib.load(open('model.joblib', 'rb'))

#     st.title("Sentiment Analysis")
#     st.text("Basic app to detect the sentiment of text.")
#     st.text("")
#     userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
#     st.text("")
#     predicted_sentiment = ""
#     if st.button("Predict"):
#         predicted_sentiment = model.predict(pd.Series(userinput))[0]
#         if predicted_sentiment == 1:
#             output = 'positive 👍'
#         else:
#             output = 'negative 👎'
#         sentiment=f'Predicted sentiment of "{userinput}" is {output}.'
#         st.success(sentiment)

# if __name__ == "__main__":

import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    model = joblib.load(open('model.joblib', 'r'))

    st.title("🌍 Language Detector")
    st.text("Basic app to detect the language of a sentence.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Detect button.', placeholder='Input text HERE')
    st.text("")
    detected_language = ""
    if st.button("Detect"):
        cleaned_input = preprocessor(userinput)
        detected_language = model.predict(pd.Series(cleaned_input))[0]
        st.success(f'The detected language of "{userinput}" is **{detected_language.upper()}**.')

if __name__ == "__main__":
    run()

#     run()
