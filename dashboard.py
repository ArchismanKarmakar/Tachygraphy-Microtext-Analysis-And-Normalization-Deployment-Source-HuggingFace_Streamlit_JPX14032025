import streamlit as st
from transformers.utils.hub import TRANSFORMERS_CACHE
import shutil
import torch
import psutil
import gc
import os

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


def create_footer():
    st.divider()

    # 🛠️ Layout using Streamlit columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # 🚀 Contributors Section
    with col1:
        st.markdown("### 🚀 Contributors")
        st.write("**Archisman Karmakar**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/archismankarmakar/) | [🐙 GitHub](https://www.github.com/ArchismanKarmakar) | [📊 Kaggle](https://www.kaggle.com/archismancoder)")

        st.write("**Sumon Chatterjee**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/sumon-chatterjee-3b3b43227) | [🐙 GitHub](https://github.com/Sumon670) | [📊 Kaggle](https://www.kaggle.com/sumonchatterjee)")

    # 🎓 Mentors Section
    with col2:
        st.markdown("### 🎓 Mentors")
        st.write("**Prof. Anupam Mondal**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/anupam-mondal-ph-d-8a7a1a39/) | [📚 Google Scholar](https://scholar.google.com/citations?user=ESRR9o4AAAAJ&hl=en) | [🌐 Website](https://sites.google.com/view/anupammondal/home)")

        st.write("**Prof. Sainik Kumar Mahata**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/mahatasainikk) | [📚 Google Scholar](https://scholar.google.co.in/citations?user=OcJDM50AAAAJ&hl=en) | [🌐 Website](https://sites.google.com/view/sainik-kumar-mahata/home)")

    # 📌 Research Project Info Section
    with col3:
        st.markdown("### 📝 About the Project")
        st.write("This is our research project for our **B.Tech final year** and a **journal** which is yet to be published.")
        st.write("Built with 💙 using **Streamlit**.")

# 🚀 Display Footer

def show_dashboard():
    # free_memory()
    st.title("Tachygraphy Micro-text Analysis & Normalization")
    st.write("""
        Welcome to the Tachygraphy Micro-text Analysis & Normalization Project. This application is designed to analyze text data through three stages:
        1. Sentiment Polarity Analysis
        2. Emotion Mood-tag Analysis
        3. Text Transformation & Normalization
    """)

    create_footer()


def __main__():
    show_dashboard()