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

def show_dashboard():
    # free_memory()
    st.title("Tachygraphy Micro-text Analysis & Normalization")
    st.write("""
        Welcome to the Tachygraphy Micro-text Analysis & Normalization Project. This application is designed to analyze text data through three stages:
        1. Sentiment Polarity Analysis
        2. Emotion Mood-tag Analysis
        3. Text Transformation & Normalization
    """)


def __main__():
    show_dashboard()