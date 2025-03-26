import shutil
from transformers.utils.hub import TRANSFORMERS_CACHE
import torch
import time
import joblib
import importlib.util
from imports import *
import os
import sys
import time
import uuid
import math

from dotenv import load_dotenv
# import psycopg2
from supabase import create_client, Client
from datetime import datetime, timezone
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

env_path = os.path.join(os.path.dirname(__file__),
                        "..", ".devcontainer", ".env")

# from transformers.utils import move_cache_to_trash
# from huggingface_hub import delete_cache


# from hmv_cfg_base_stage1.model1 import load_model as load_model1
# from hmv_cfg_base_stage1.model1 import predict as predict1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "sentimentPolarity_analysis", "config", "stage1_models.json")
CONFIG_STAGE2 = os.path.join(BASE_DIR, "..", "emotionMoodtag_analysis", "config", "stage2_models.json")
CONFIG_STAGE3 = os.path.join(BASE_DIR, "..", "transformation_and_Normalization", "config", "stage3_models.json")
LOADERS_STAGE_COLLECTOR = os.path.join(BASE_DIR, "hmv_cfg_base_dlc")


EMOTION_MOODTAG_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral",
    "sadness", "surprise"
]

SENTIMENT_POLARITY_LABELS = [
    "negative", "neutral", "positive"
]


current_model = None
current_tokenizer = None


# Enabling Resource caching

# Load environment variables from .env
load_dotenv()

# @st.cache_resource
# DATABASE_URL = os.environ.get("DATABASE_URL")

# def get_connection():
#     #  """Establish a connection to the database."""
#     # return psycopg2.connect(os.environ.get("DATABASE_URL"))
#     supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("anon_key"))
#     return supabase

# @st.cache_resource


def load_model_config1():
    with open(CONFIG_STAGE1, "r") as f:
        model_data = json.load(f)

    # Extract names for dropdown
    # model_options is a dict mapping model name to its config
    model_options = {v["name"]: v for v in model_data.values()}

    # Create an OrderedDict and insert a default option at the beginning.
    default_option = "--Select the model used for inference (if applicable)--"
    model_options_with_default = OrderedDict()
    model_options_with_default[default_option] = None  # or any placeholder value
    # Add the rest of the options
    for key, value in model_options.items():
        model_options_with_default[key] = value

    return model_data, model_options_with_default


MODEL_DATA1, MODEL_OPTIONS1 = load_model_config1()


def load_model_config2():
    with open(CONFIG_STAGE2, "r") as f:
        model_data = json.load(f)

    # Extract names for dropdown
    # model_options is a dict mapping model name to its config
    model_options = {v["name"]: v for v in model_data.values()}

    # Create an OrderedDict and insert a default option at the beginning.
    default_option = "--Select the model used for inference (if applicable)--"
    model_options_with_default = OrderedDict()
    model_options_with_default[default_option] = None  # or any placeholder value
    # Add the rest of the options
    for key, value in model_options.items():
        model_options_with_default[key] = value

    return model_data, model_options_with_default

MODEL_DATA2, MODEL_OPTIONS2 = load_model_config2()


def load_model_config3():
    with open(CONFIG_STAGE3, "r") as f:
        model_data = json.load(f)

    # Extract names for dropdown
    # model_options is a dict mapping model name to its config
    model_options = {v["name"]: v for v in model_data.values()}

    # Create an OrderedDict and insert a default option at the beginning.
    default_option = "--Select the model used for inference (if applicable)--"
    model_options_with_default = OrderedDict()
    model_options_with_default[default_option] = None  # or any placeholder value
    # Add the rest of the options
    for key, value in model_options.items():
        model_options_with_default[key] = value

    return model_data, model_options_with_default


MODEL_DATA3, MODEL_OPTIONS3 = load_model_config3()


# ✅ Dynamically Import Model Functions
def import_from_module(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ModuleNotFoundError, AttributeError) as e:
        st.error(f"❌ Import Error: {e}")
        return None


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


def disable_ui():
    st.components.v1.html(
        """
        <style>
        #ui-disable-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(200, 200, 200, 0.5);
            z-index: 9999;
        }
        </style>
        <div id="ui-disable-overlay"></div>
        """,
        height=0,
        scrolling=False
    )


def enable_ui():
    st.components.v1.html(
        """
        <script>
        var overlay = document.getElementById("ui-disable-overlay");
        if (overlay) {
            overlay.parentNode.removeChild(overlay);
        }
        </script>
        """,
        height=0,
        scrolling=False
    )

# Function to increment progress dynamically


def get_env_variable(var_name):
    # Try os.environ first (this covers local development and HF Spaces)
    value = os.environ.get(var_name)
    if value is None:
        # Fall back to st.secrets if available (e.g., on Streamlit Cloud)
        try:
            value = st.secrets[var_name]
        except KeyError:
            value = None
    return value


def show_data_collector():
    st.title("Data Correction & Collection Page")

    st.error("New API keys are coming in Q2 2025, May 1st, old API authentication will be deprecated and blocked by PostgREST.")
    st.warning(
        "This page is running in test mode, please be careful with your data.")
    st.error("The database is running in debug log mode, please be careful with your data.")

    with st.form("feedback_form", clear_on_submit=True, border=False):
        st.write("### Data Collection Form")
        st.write(
            "#### If the predictions generated are wrong, please provide feedback to help improve the model.")

        # Model selection dropdown for Stage 3
        model_names3 = list(MODEL_OPTIONS3.keys())
        selected_model3 = st.selectbox(
            "Choose a model:", model_names3, key="selected_model_stage3"
        )

        # Text Feedback Inputs
        col1, col2 = st.columns(2)
        with col1:
            feedback = st.text_input(
                "Enter the correct / actual expanded standard formal English text:",
                key="feedback_input"
            )
        with col2:
            feedback2 = st.text_input(
                "Enter any one of the wrongly predicted text:",
                key="feedback_input2"
            )

        st.warning(
        "The correct slider is for the actual probability of the label and wrong slider is the predicted probability by any model which is wrong for that label.")


            
        st.write("#### Sentiment Polarity Probabilities (Select values between 0 and 1)")
        SENTIMENT_POLARITY_LABELS = ["negative", "neutral", "positive"]

        model_names1 = list(MODEL_OPTIONS1.keys())
        selected_model1 = st.selectbox(
            "Choose a model:", model_names1, key="selected_model_stage1"
        )

        sentiment_feedback = {}
        # For sentiment, we have 3 labels so we can place them in one row.
        sentiment_cols = st.columns(len(SENTIMENT_POLARITY_LABELS))
        for idx, label in enumerate(SENTIMENT_POLARITY_LABELS):
            with sentiment_cols[idx]:
                st.write(f"##### **{label.capitalize()}**")
                # Create two subcolumns for "Correct" and "Wrong"
                subcol_correct, subcol_wrong = st.columns(2)
                with subcol_correct:
                    correct_value = st.slider(
                        "Correct",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.33,  # default value
                        step=0.01,
                        format="%.2f",
                        key=f"sentiment_{label}_correct"
                    )
                with subcol_wrong:
                    wrong_value = st.slider(
                        "Wrong",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,   # default value
                        step=0.01,
                        format="%.2f",
                        key=f"sentiment_{label}_wrong"
                    )
            sentiment_feedback[label] = {"correct": correct_value, "wrong": wrong_value}

        # st.write("**Collected Sentiment Feedback:**")
        # st.write(sentiment_feedback)

        # ---------------------------
        # Emotion Feedback
        # ---------------------------
        st.write("#### Emotion Probabilities (Select values between 0 and 1)")
        EMOTION_MOODTAG_LABELS = [
            "anger", "disgust", "fear", "joy", "neutral",
            "sadness", "surprise"
        ]

        model_names2 = list(MODEL_OPTIONS2.keys())
        selected_model2 = st.selectbox(
            "Choose a model:", model_names2, key="selected_model_stage2"
        )

        emotion_feedback = {}
        max_cols = 3  # Maximum number of emotion labels in one row
        num_labels = len(EMOTION_MOODTAG_LABELS)
        num_rows = math.ceil(num_labels / max_cols)

        for row in range(num_rows):
            # Get labels for this row.
            row_labels = EMOTION_MOODTAG_LABELS[row * max_cols:(row + 1) * max_cols]
            # Create main columns for each label in this row.
            main_cols = st.columns(len(row_labels))
            for idx, label in enumerate(row_labels):
                with main_cols[idx]:
                    st.write(f"##### **{label.capitalize()}**")
                    # Create two subcolumns for correct and wrong values.
                    subcol_correct, subcol_wrong = st.columns(2)
                    with subcol_correct:
                        correct_value = st.slider(
                            "Correct",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                            format="%.2f",
                            key=f"emotion_{label}_correct"
                        )
                    with subcol_wrong:
                        wrong_value = st.slider(
                            "Wrong",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                            format="%.2f",
                            key=f"emotion_{label}_wrong"
                        )
                emotion_feedback[label] = {"correct": correct_value, "wrong": wrong_value}


        # Use form_submit_button instead of st.button inside a form
        submit_feedback = st.form_submit_button("Submit Data")

        if submit_feedback and feedback.strip() and feedback2.strip():
            # Prepare data to insert
            data_to_insert = {
                "input_text": st.session_state.get("user_input_stage3", ""),
                "correct_text_by_user": feedback,
                "model_used": st.session_state.get("selected_model_stage3", "unknown"),
                "wrong_pred_any": feedback2,
                "sentiment_feedback": sentiment_feedback,
                "emotion_feedback": emotion_feedback
            }
            st.error("Submission is disabled in debug logging mode.")
            # try:
            #     from supabase import create_client, Client
            #     from dotenv import load_dotenv
            #     load_dotenv()  # or load_dotenv(dotenv_path=env_path) if you have a specific path
            #     supabase: Client = create_client(
            #         get_env_variable("SUPABASE_DB_TACHYGRAPHY_DB_URL"),
            #         get_env_variable("SUPABASE_DB_TACHYGRAPHY_ANON_API_KEY")
            #     )
            #     response = supabase.table(
            #        get_env_variable("SUPABASE_DB_TACHYGRAPHY_DB_STAGE3_TABLE")
            #     ).insert(data_to_insert, returning="minimal").execute()
            #     st.success("Feedback submitted successfully!")
            # except Exception as e:
            #     st.error(f"Feedback submission failed: {e}")
