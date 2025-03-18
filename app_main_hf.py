import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

from streamlit_extras.bottom_container import bottom
from streamlit_extras.app_logo import add_logo
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
import torch
from imports import *
import streamlit as st
from streamlit_option_menu import option_menu
import asyncio
import shutil
import gc
from transformers.utils.hub import TRANSFORMERS_CACHE

# torch.classes.__path__ = []


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

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

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))


from emotion_analysis import show_emotion_analysis
from sentiment_analysis.sentiment_analysis_main import show_sentiment_analysis
from dashboard import show_dashboard


# from text_transformation import show_text_transformation

st.set_page_config(
    page_title="Tachygraphy Microtext Analysis & Normalization",
    # layout="wide"
)


def free_memory():
    #  """Free up CPU & GPU memory before loading a new model."""
    global current_model, current_tokenizer

    if current_model is not None:
        del current_model  # Delete the existing model
        current_model = None  # Reset reference

    if current_tokenizer is not None:
        del current_tokenizer  # Delete the tokenizer
        current_tokenizer = None

    gc.collect()  # Force garbage collection for CPU memory

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free GPU memory
        torch.cuda.ipc_collect()  # Clean up PyTorch GPU cache

    # If running on CPU, reclaim memory using OS-level commands
    try:
        if torch.cuda.is_available() is False:
            psutil.virtual_memory()  # Refresh memory stats
    except Exception as e:
        print(f"Memory cleanup error: {e}")

    # Delete cached Hugging Face models
    try:
        cache_dir = TRANSFORMERS_CACHE
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Cache cleared!")
    except Exception as e:
        print(f"❌ Cache cleanup error: {e}")



def main():
    # selection = option_menu(
    #     menu_title="Navigation",
    #     options=[
    #         "Dashboard",
    #         "Stage 1: Sentiment Polarity Analysis",
    #         "Stage 2: Emotion Mood-tag Analysis",
    #         "Stage 3: Text Transformation & Normalization"
    #     ],
    #     icons=["joystick", "bar-chart", "emoji-laughing", "pencil"],
    #     styles={
    #         "container": {}},
    #     menu_icon="menu-button-wide-fill",
    #     default_index=0,
    #     orientation="horizontal"
    # )

    st.sidebar.title("Navigation")
    with st.sidebar:
        selection = option_menu(
            menu_title=None,          # No title for a sleek look
            options=["Dashboard", "Stage 1: Sentiment Polarity Analysis", "Stage 2: Emotion Mood-tag Analysis", "Stage 3: Text Transformation & Normalization"],
            icons=None,
            menu_icon="cast",          # Main menu icon
            default_index=0,           # Highlight the first option
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "#6c757d", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "color": "#6c757d",
                    "transition": "0.3s",
                },
                "nav-link-selected": {
                    "background-color": "#FF4B4B",
                    "color": "white",
                    "font-weight": "bold",
                    "border-radius": "8px",
                },
            }
        )

    # st.sidebar.title("Navigation")
    # selection = st.sidebar.radio("Go to", ["Dashboard", "Stage 1: Sentiment Polarity Analysis", "Stage 2: Emotion Mood-tag Analysis", "Stage 3: Text Transformation & Normalization"])

    # if selection == "Dashboard":
    #     show_dashboard()
    # elif selection == "Stage 1: Sentiment Polarity Analysis":
    #     show_sentiment_analysis()
    # elif selection == "Stage 2: Emotion Mood-tag Analysis":
    #     # show_emotion_analysis()
    #     st.write("This section is under development.")
    # elif selection == "Stage 3: Text Transformation & Normalization":
    #     # show_text_transformation()
    #     st.write("This section is under development.")

    if selection == "Dashboard":
        st.cache_resource.clear()
        # free_memory()
        show_dashboard()

    elif selection == "Stage 1: Sentiment Polarity Analysis":
        st.cache_resource.clear()
        # free_memory()
        show_sentiment_analysis()

    elif selection == "Stage 2: Emotion Mood-tag Analysis":
        st.cache_resource.clear()
        # free_memory()
        # show_emotion_analysis()
        st.write("This section is under development.")

    elif selection == "Stage 3: Text Transformation & Normalization":
        st.cache_resource.clear()
        # free_memory()
        # show_text_transformation()
        st.write("This section is under development.")



    # st.sidebar.title("Navigation")
    # selection = st.sidebar.radio("Go to", ["Dashboard", "Stage 1: Sentiment Polarity Analysis", "Stage 2: Emotion Mood-tag Analysis", "Stage 3: Text Transformation & Normalization"])

    # if selection == "Dashboard":
    #     show_dashboard()
    # elif selection == "Stage 1: Sentiment Polarity Analysis":
    #     show_sentiment_analysis()
    # elif selection == "Stage 2: Emotion Mood-tag Analysis":
    #     # show_emotion_analysis()
    #     st.write("This section is under development.")
    # elif selection == "Stage 3: Text Transformation & Normalization":
    #     # show_text_transformation()
    #     st.write("This section is under development.")

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
