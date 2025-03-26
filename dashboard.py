import gc
import psutil
import torch
import shutil
from transformers.utils.hub import TRANSFORMERS_CACHE
import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))


def free_memory():
    #  """Free up CPU & GPU memory before loading a new model."""
    # global current_model, current_tokenizer

    # if current_model is not None:
    #     del current_model  # Delete the existing model
    #     current_model = None  # Reset reference

    # if current_tokenizer is not None:
    #     del current_tokenizer  # Delete the tokenizer
    #     current_tokenizer = None

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


def create_sample_example3():
    st.write("""
        #### Sample Example 3
        """)
    graph = """
    digraph {
        // Global graph settings with explicit DPI
        graph [bgcolor="white", rankdir=TB, splines=true, nodesep=0.8, ranksep=0.8];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.15,0.1"];

        // Define nodes with custom colors
        Input [label="Input:\nbruh, floods in Kerala, rescue ops non-stop 🚁", fillcolor="#ffe6de", fontcolor="#000000"];
        Output [label="Output:\nBrother, the floods in Kerala are severe,\nand rescue operations are ongoing continuously.", fillcolor="#ffe6de", fontcolor="#000000"];
        Sentiment [label="Sentiment:\nNEUTRAL", fillcolor="#ecdeff", fontcolor="black"];

        // Emotion nodes with a uniform style
        Anger [label="Anger: 0.080178231", fillcolor="#deffe1", fontcolor="black"];
        Disgust [label="Disgust: 0.015257259", fillcolor="#deffe1", fontcolor="black"];
        Fear [label="Fear: 0.601871967", fillcolor="#deffe1", fontcolor="black"];
        Joy [label="Joy: 0.00410547", fillcolor="#deffe1", fontcolor="black"];
        Neutral [label="Neutral: 0.0341026", fillcolor="#deffe1", fontcolor="black"];
        Sadness [label="Sadness: 0.245294735", fillcolor="#deffe1", fontcolor="black"];
        Surprise [label="Surprise: 0.019189769", fillcolor="#deffe1", fontcolor="black"];

        // Define edges with a consistent style
        edge [color="#7a7a7a", penwidth=3];

        // Establish the tree structure
        Input -> Output;
        Input -> Sentiment;
        Sentiment -> Emotion
        Emotion -> Anger;
        Emotion -> Disgust;
        Emotion -> Fear;
        Emotion -> Joy;
        Emotion -> Neutral;
        Emotion -> Sadness;
        Emotion -> Surprise;
    }
    """
    st.graphviz_chart(graph)


def create_sample_example2():
    st.write("""
        #### Sample Example 2
        """)
    graph = """
    digraph {
        // Global graph settings
        graph [bgcolor="white", rankdir=TB, splines=true, nodesep=0.8, ranksep=0.8];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.15,0.1"];

        // Define nodes with custom colors
        Input [label="Input:\nu rlly think all that talk means u tough? lol, when I step up, u ain't gon say sh*t", fillcolor="#ffe6de", fontcolor="black"];
        Output [label="Output:\nyou really think all that talk makes you tough [lol](laughed out loud) when i step up you are not going to say anything", fillcolor="#ffe6de", fontcolor="black"];
        Sentiment [label="Sentiment:\nNEGATIVE", fillcolor="#ecdeff", fontcolor="black"];

        // Emotion nodes with a uniform style
        Anger [label="Anger: 0.14403291", fillcolor="#deffe1", fontcolor="black"];
        Disgust [label="Disgust: 0.039282672", fillcolor="#deffe1", fontcolor="black"];
        Fear [label="Fear: 0.014349542", fillcolor="#deffe1", fontcolor="black"];
        Joy [label="Joy: 0.048965044", fillcolor="#deffe1", fontcolor="black"];
        Neutral [label="Neutral: 0.494852662", fillcolor="#deffe1", fontcolor="black"];
        Sadness [label="Sadness: 0.021111647", fillcolor="#deffe1", fontcolor="black"];
        Surprise [label="Surprise: 0.237405464", fillcolor="#deffe1", fontcolor="black"];

        // Define edges with a consistent style
        edge [color="#7a7a7a", penwidth=3];

        // Establish the tree structure
        Input -> Output;
        Input -> Sentiment;
        Sentiment -> Emotion
        Emotion -> Anger;
        Emotion -> Disgust;
        Emotion -> Fear;
        Emotion -> Joy;
        Emotion -> Neutral;
        Emotion -> Sadness;
        Emotion -> Surprise;
    }
    """
    st.graphviz_chart(graph)


def create_sample_example1():
    st.write("#### Sample Example 1")

    graph = """
    digraph G {
        rankdir=LR;
        bgcolor="white";
        nodesep=0.8;
        ranksep=0.8;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.15,0.1"];

        // Define nodes with colors
        "Input Text" [label="Input Text:\ni don't know for real y he's sooo sad", fillcolor="#ffe6de", fontcolor="black"];
        "Normalized Text" [label="Normalized Text:\ni do not know for real why he's so sad", fillcolor="#e6f4d7", fontcolor="black"];
        "Sentiment" [label="Sentiment", fillcolor="#fde6ff", fontcolor="black"];
        "negative" [label="negative: 0.995874803543091", fillcolor="#e8e6ff", fontcolor="black"];
        "neutral" [label="neutral: 6.232635259628296e-05", fillcolor="#e8e6ff", fontcolor="black"];
        "positive" [label="positive: 2.0964847564697266e-05", fillcolor="#e8e6ff", fontcolor="black"];

        "Emotion" [label="Emotion", fillcolor="#fdf5e6", fontcolor="black"];
        "anger" [label="anger: 0.0", fillcolor="#deffe1", fontcolor="black"];
        "disgust" [label="disgust: 0.0", fillcolor="#deffe1", fontcolor="black"];
        "fear" [label="fear: 0.010283803842246056", fillcolor="#deffe1", fontcolor="black"];
        "joy" [label="joy: 0.0", fillcolor="#deffe1", fontcolor="black"];
        "neutral_e" [label="neutral: 0.021935827255129814", fillcolor="#deffe1", fontcolor="black"];
        "sadness" [label="sadness: 1.0", fillcolor="#deffe1", fontcolor="black"];
        "surprise" [label="surprise: 0.02158345977962017", fillcolor="#deffe1", fontcolor="black"];

        // Define edges
        "Input Text" -> "Normalized Text";
        "Normalized Text" -> "Sentiment";
        "Sentiment" -> "negative";
        "Sentiment" -> "neutral";
        "Sentiment" -> "positive";

        "Normalized Text" -> "Emotion";
        "Emotion" -> "anger";
        "Emotion" -> "disgust";
        "Emotion" -> "fear";
        "Emotion" -> "joy";
        "Emotion" -> "neutral_e";
        "Emotion" -> "sadness";
        "Emotion" -> "surprise";
    }
    """

    st.graphviz_chart(graph)



def create_project_overview():
    # st.divider()
    st.markdown("## Project Overview")
    st.write(f"""
        Tachygraphy—originally developed to expedite writing—has evolved over centuries. In the 1990s, it reappeared as micro-text, driving faster communication on social media with characteristics like 'Anytime, Anyplace, Anybody, and Anything (4A)'. This project focuses on the analysis and normalization of micro-text, which is a prevalent form of informal communication today. It aims to enhance Natural Language Processing (NLP) tasks by standardizing micro-text for better sentiment analysis, emotion analysis, data extraction and normalization to understandable form aka. 4A message decoding as primary objective.
        """
             )


def create_footer():
    # st.divider()
    st.markdown("## About Us")

    # 🛠️ Layout using Streamlit columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # 🚀 Contributors Section
    with col1:
        st.markdown("### 🚀 Contributors")
        st.write("##### **Archisman Karmakar**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/archismankarmakar/) | [🐙 GitHub](https://www.github.com/ArchismanKarmakar) | [📊 Kaggle](https://www.kaggle.com/archismancoder)")

        st.write("##### **Sumon Chatterjee**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/sumon-chatterjee-3b3b43227) | [🐙 GitHub](https://github.com/Sumon670) | [📊 Kaggle](https://www.kaggle.com/sumonchatterjee)")

    # 🎓 Mentors Section
    with col2:
        st.markdown("### 🎓 Mentors")
        st.write("##### **Prof. Anupam Mondal**")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/anupam-mondal-ph-d-8a7a1a39/) | [📚 Google Scholar](https://scholar.google.com/citations?user=ESRR9o4AAAAJ&hl=en) | [🌐 Website](https://sites.google.com/view/anupammondal/home)")

        st.write("##### **Prof. Sainik Kumar Mahata**")
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
    st.write(f"""Welcome to the Tachygraphy Micro-text Analysis & Normalization Project. This application is designed to analyze text data through three stages:""")
    coltl1, coltl2 = st.columns(2)
    with coltl1:
        st.write("""
            1. Sentiment Polarity Analysis
            2. Emotion Mood-tag Analysis
            3. Text Transformation & Normalization
            4. Stacked all 3 stages with their best models
            5. Data Correction & Collection
        """)
    with coltl2:
        st.write("""
                - Training Source: [GitHub @ Tachygraphy Micro-text Analysis & Normalization](https://github.com/ArchismanKarmakar/Tachygraphy-Microtext-Analysis-And-Normalization)
                - Kaggle Collections: [Kaggle @ Tachygraphy Micro-text Analysis & Normalization](https://www.kaggle.com/datasets/archismancoder/dataset-tachygraphy/data?select=Tachygraphy_MicroText-AIO-V3.xlsx)
                - Hugging Face Org: [Hugging Face @ Tachygraphy Micro-text Analysis & Normalization](https://huggingface.co/Tachygraphy-Microtext-Normalization-IEMK25)
                - Deployment Source: [GitHub](https://github.com/ArchismanKarmakar/Tachygraphy-Microtext-Analysis-And-Normalization-Deployment-Source-HuggingFace_Streamlit_JPX14032025)
                - Streamlit Deployemnt: [Streamlit](https://tachygraphy-microtext.streamlit.app/)
                - Hugging Face Space Deployment: [Hugging Face Space](https://huggingface.co/spaces/Tachygraphy-Microtext-Normalization-IEMK25/Tachygraphy-Microtext-Analysis-and-Normalization-ArchismanCoder)
                """)

    create_footer()

    create_project_overview()


    create_sample_example1()

    create_sample_example2()
    create_sample_example3()


def __main__():
    show_dashboard()
