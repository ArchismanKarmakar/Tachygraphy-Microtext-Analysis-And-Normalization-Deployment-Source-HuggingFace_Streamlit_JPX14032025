import streamlit as st
import os
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


import joblib
import importlib
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

from imports import *




from dashboard import show_dashboard
from sentiment_analysis.sentiment_analysis_main import show_sentiment_analysis
from emotion_analysis import show_emotion_analysis
# from text_transformation import show_text_transformation


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Dashboard", "Stage 1: Sentiment Polarity Analysis", "Stage 2: Emotion Mood-tag Analysis", "Stage 3: Text Transformation & Normalization"])

    if selection == "Dashboard":
        show_dashboard()
    elif selection == "Stage 1: Sentiment Polarity Analysis":
        show_sentiment_analysis()
    elif selection == "Stage 2: Emotion Mood-tag Analysis":
        # show_emotion_analysis()
        st.write("This section is under development.")
    elif selection == "Stage 3: Text Transformation & Normalization":
        # show_text_transformation()
        st.write("This section is under development.")

    st.sidebar.title("About")
    st.sidebar.info("""
        **Contributors:**
        - Archisman Karmakar 
            - [LinkedIn](https://www.linkedin.com/in/archismankarmakar/)
            - [GitHub](https://www.github.com/ArchismanKarmakar)
            - [Kaggle](https://www.kaggle.com/archismancoder)
        - Sumon Chatterjee
            - [LinkedIn](https://www.linkedin.com/in/sumon-chatterjee-3b3b43227)
            - [GitHub](https://github.com/Sumon670)
            - [Kaggle](https://www.kaggle.com/sumonchatterjee)

        **Mentors:**
        - Prof. Anupam Mondal
            - [LinkedIn](https://www.linkedin.com/in/anupam-mondal-ph-d-8a7a1a39/)
            - [Google Scholar](https://scholar.google.com/citations?user=ESRR9o4AAAAJ&hl=en)
            - [Website](https://sites.google.com/view/anupammondal/home)
        - Prof. Sainik Kumar Mahata
            - [LinkedIn](https://www.linkedin.com/in/mahatasainikk)
            - [Google Scholar](https://scholar.google.co.in/citations?user=OcJDM50AAAAJ&hl=en)
            - [Website](https://sites.google.com/view/sainik-kumar-mahata/home)

        This is our research project for our B.Tech final year and a journal which is yet to be published.
    """)

if __name__ == "__main__":
    main()