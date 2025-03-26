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
    # Convert model_data values to a list and take only the first two entries
    top2_data = list(model_data.values())[:2]
    # Create a dictionary mapping from model name to its configuration for the top two models
    model_options = {v["name"]: v for v in top2_data}
    return top2_data, model_options



MODEL_DATA1, MODEL_OPTIONS1 = load_model_config1()

# MODEL_DATA1_1=MODEL_DATA1[0]
# MODEL_OPTIONS1_1=MODEL_OPTIONS1[0]


def load_model_config2():
    with open(CONFIG_STAGE2, "r") as f:
        model_data = json.load(f)
    # Convert model_data values to a list and take only the first two entries
    top2_data = list(model_data.values())[:2]
    # Create a dictionary mapping from model name to its configuration for the top two models
    model_options = {v["name"]: v for v in top2_data}
    return top2_data, model_options


MODEL_DATA2, MODEL_OPTIONS2 = load_model_config2()

# MODEL_DATA2_1=MODEL_DATA2[0]
# MODEL_OPTIONS2_1=MODEL_OPTIONS2[0]


def load_model_config3():
    with open(CONFIG_STAGE3, "r") as f:
        model_data = json.load(f)
    # Convert model_data values to a list and take only the first two entries
    top2_data = list(model_data.values())[:2]
    # Create a dictionary mapping from model name to its configuration for the top two models
    model_options = {v["name"]: v for v in top2_data}
    return top2_data, model_options



MODEL_DATA3, MODEL_OPTIONS3 = load_model_config3()

# MODEL_DATA3_1=MODEL_DATA3[0]
# MODEL_OPTIONS3_1=MODEL_OPTIONS3[0]


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


def load_selected_model1(model_name):
    global current_model, current_tokenizer

    # st.cache_resource.clear()

    # free_memory()

    # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
    # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[model_name])  # ✅ Check selected model
    # st.write("DEBUG: Model Name:", model_name)  # ✅ Check selected model

    if model_name not in MODEL_OPTIONS1:
        st.error(f"⚠️ Model '{model_name}' not found in config!")
        return None, None, None

    model_info = MODEL_OPTIONS1[model_name]
    hf_location = model_info["hf_location"]

    model_module = model_info["module_path"]
    load_function = model_info["load_function"]
    predict_function = model_info["predict_function"]

    load_model_func = import_from_module(model_module, load_function)
    predict_func = import_from_module(model_module, predict_function)

    if load_model_func is None or predict_func is None:
        st.error("❌ Model functions could not be loaded!")
        return None, None, None

    model, tokenizer = load_model_func()

    current_model, current_tokenizer = model, tokenizer
    return model, tokenizer, predict_func

def load_selected_model2(model_name):
    global current_model, current_tokenizer

    # st.cache_resource.clear()

    # free_memory()

    # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
    # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[model_name])  # ✅ Check selected model
    # st.write("DEBUG: Model Name:", model_name)  # ✅ Check selected model

    if model_name not in MODEL_OPTIONS2:
        st.error(f"⚠️ Model '{model_name}' not found in config!")
        return None, None, None

    model_info = MODEL_OPTIONS2[model_name]
    hf_location = model_info["hf_location"]

    model_module = model_info["module_path"]
    load_function = model_info["load_function"]
    predict_function = model_info["predict_function"]

    load_model_func = import_from_module(model_module, load_function)
    predict_func = import_from_module(model_module, predict_function)

    if load_model_func is None or predict_func is None:
        st.error("❌ Model functions could not be loaded!")
        return None, None, None

    model, tokenizer = load_model_func()

    current_model, current_tokenizer = model, tokenizer
    return model, tokenizer, predict_func

def load_selected_model3(model_name):
    global current_model, current_tokenizer

    # st.cache_resource.clear()

    # free_memory()

    # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
    # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[model_name])  # ✅ Check selected model
    # st.write("DEBUG: Model Name:", model_name)  # ✅ Check selected model

    if model_name not in MODEL_OPTIONS3:
        st.error(f"⚠️ Model '{model_name}' not found in config!")
        return None, None, None

    model_info = MODEL_OPTIONS3[model_name]
    hf_location = model_info["hf_location"]

    model_module = model_info["module_path"]
    load_function = model_info["load_function"]
    predict_function = model_info["predict_function"]

    load_model_func = import_from_module(model_module, load_function)
    predict_func = import_from_module(model_module, predict_function)

    if load_model_func is None or predict_func is None:
        st.error("❌ Model functions could not be loaded!")
        return None, None, None

    model, tokenizer = load_model_func()

    current_model, current_tokenizer = model, tokenizer
    return model, tokenizer, predict_func


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


def get_sentiment_emotion_graph_code(input_text, normalized_text, sentiment_array, emotion_array):
    """
    Returns a Graphviz code string representing:
      - Input Text as the root
      - Normalized Text as a child
      - A Sentiment node with its probabilities as children (using SENTIMENT_POLARITY_LABELS)
      - An Emotion node with its probabilities as children (using EMOTION_MOODTAG_LABELS)
      - Arrows from each sentiment node to the Emotion node with fixed penwidths (5 for highest, 3 for middle, 1 for lowest)
      
    Both sentiment_array and emotion_array are NumPy arrays (possibly nested, e.g. [[values]]),
    so they are squeezed before use.
    """
    import numpy as np

    # Flatten arrays in case they are nested
    sentiment_flat = np.array(sentiment_array).squeeze()
    emotion_flat = np.array(emotion_array).squeeze()

    # Create pairs for each sentiment label with its probability
    sentiment_pairs = list(zip(SENTIMENT_POLARITY_LABELS, sentiment_flat))
    # Sort by probability (ascending)
    sentiment_sorted = sorted(sentiment_pairs, key=lambda x: x[1])
    
    # Create a penwidth map: label -> penwidth
    penwidth_map = {}
    
    # Collect all unique probabilities to handle ties
    unique_probs = set(prob for _, prob in sentiment_sorted)
    
    if len(unique_probs) == 1:
        # All sentiments have the same probability; use mid-range width (e.g., 3) for all
        for label, _ in sentiment_sorted:
            penwidth_map[label] = 3
    elif len(unique_probs) == 2:
        # Two unique probabilities: assign min width 1 and max width 5 accordingly
        min_prob = sentiment_sorted[0][1]
        max_prob = sentiment_sorted[-1][1]
        for label, prob in sentiment_sorted:
            if prob == min_prob:
                penwidth_map[label] = 1
            else:
                penwidth_map[label] = 5
    else:
        # For three distinct probabilities, assign 1 to the smallest, 3 to the middle, 5 to the largest.
        penwidth_map[sentiment_sorted[0][0]] = 0.2
        penwidth_map[sentiment_sorted[1][0]] = 0.3
        penwidth_map[sentiment_sorted[2][0]] = 1

    # Build the basic Graphviz structure
    graph_code = f'''
    digraph G {{
        rankdir=TB;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
        
        Input [label="Input Text:\\n{input_text.replace('"', '\\"')}", fillcolor="#ffe6de", fontcolor="#000000"];
        Normalized [label="Normalized Text:\\n{normalized_text.replace('"', '\\"')}", fillcolor="#f8ffde", fontcolor="#000000"];
        Sentiment [label="Sentiment", fillcolor="#ffefde", fontcolor="black"];
        Emotion [label="Emotion", fillcolor="#ffefde", fontcolor="black"];
        
        Input -> Normalized;
        Input -> Sentiment;
        Sentiment -> Emotion;
    '''
    
    # Add sentiment nodes (displaying full values without truncation)
    for label, prob in sentiment_pairs:
        node_id = f"S_{label}"
        graph_code += f'\n    {node_id} [label="{label}: {prob}", fillcolor="#f6edfc", fontcolor="black"];'
        graph_code += f'\n    Sentiment -> {node_id};'
    
    # Add emotion nodes (displaying full values)
    for i, label in enumerate(EMOTION_MOODTAG_LABELS):
        if i < len(emotion_flat):
            prob = emotion_flat[i]
            node_id = f"E_{label}"
            graph_code += f'\n    {node_id} [label="{label}: {prob}", fillcolor="#edfcef", fontcolor="black"];'
            graph_code += f'\n    Emotion -> {node_id};'
    
    # Add arrows from each sentiment node to the Emotion node with fixed penwidth based on ranking
    for label, prob in sentiment_pairs:
        node_id = f"S_{label}"
        pw = penwidth_map[label]
        graph_code += f'\n    {node_id} -> Emotion [penwidth={pw}];'

    graph_code += "\n}"
    return graph_code






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


def update_progress(progress_bar, start, end, delay=0.1):
    for i in range(start, end + 1, 5):  # Increment in steps of 5%
        progress_bar.progress(i)
        time.sleep(delay)  # Simulate processing time
        # st.experimental_rerun() # Refresh the page


# Function to update session state when model changes
def on_model_change():
    st.cache_data.clear()
    st.cache_resource.clear()
    free_memory()
    st.session_state.model_changed = True  # Mark model as changed

    # Reset flags to trigger new prediction and show feedback form
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    st.session_state.last_processed_input = ""


# Function to update session state when text changes


def on_text_change():
    st.session_state.text_changed = True  # Mark text as changed

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""


def update_top_k_from_slider():
    st.session_state.top_k = st.session_state.top_k_slider

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""


def update_top_k_from_input():
    st.session_state.top_k = st.session_state.top_k_input

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_temperature_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_top_p_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_beam_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_enable_sampling_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_enable_earlyStopping_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_max_new_tokens_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

def on_num_return_sequences_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    st.session_state.graphviz_code = None
    # st.session_state.last_processed_input = ""

# Initialize session state variables
if "selected_model1" not in st.session_state:
    st.session_state.selected_model1 = list(MODEL_OPTIONS1.keys())[
        0]  # Default model
if "selected_model2" not in st.session_state:
    st.session_state.selected_model2 = list(MODEL_OPTIONS2.keys())[
        0]
if "selected_model3" not in st.session_state:
    st.session_state.selected_model3 = list(MODEL_OPTIONS3.keys())[
        0]
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = ""
if "model_changed" not in st.session_state:
    st.session_state.model_changed = False
if "text_changed" not in st.session_state:
    st.session_state.text_changed = False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 50


if "last_change" not in st.session_state:
    st.session_state.last_change = time.time()
if "auto_predict_triggered" not in st.session_state:
    st.session_state.auto_predict_triggered = False





def show_stacking_stages():
    # No cache clearing here—only in the model change callback!

    # st.write(st.session_state)

    if "last_change" not in st.session_state:
        st.session_state.last_change = time.time()
    if "auto_predict_triggered" not in st.session_state:
        st.session_state.auto_predict_triggered = False

    
    if "top_k" not in st.session_state:
        st.session_state.top_k = 50

    model_names1 = list(MODEL_OPTIONS1.keys())
    model_names2 = list(MODEL_OPTIONS2.keys())
    model_names3 = list(MODEL_OPTIONS3.keys())

    st.title("Stacking all the best models together")

    st.warning("If memory is low, this page may take a while to load or might fail too if memory overshoots or due to CUDA_Side_Device_Assertions.")

    # Check if the stored selected model is valid; if not, reset it
    if "selected_model1" in st.session_state:
        if st.session_state.selected_model1 not in model_names1:
            st.session_state.selected_model1 = model_names1[0]
    else:
        st.session_state.selected_model1 = model_names1[0]

    if "selected_model2" in st.session_state:
        if st.session_state.selected_model2 not in model_names2:
            st.session_state.selected_model2 = model_names2[0]
    else:
        st.session_state.selected_model2 = model_names2[0]
    
    if "selected_model3" in st.session_state:
        if st.session_state.selected_model3 not in model_names3:
            st.session_state.selected_model3 = model_names3[0]
    else:
        st.session_state.selected_model3 = model_names3[0]

    # st.title("Stacking all the best models together")
    st.write("This section handles the sentiment analysis and emotion analysis of informal text and then transformation and normalization of it into standard formal English.")

    # Model selection with change detection; clearing cache happens in on_model_change()
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_model1 = st.selectbox(
            "Choose a sentiment polarity prediction model:", model_names1, key="selected_model_stage1", on_change=on_model_change
        )
    with col2:
        selected_model2 = st.selectbox(
            "Choose a emotion mood-tag prediction model:", model_names2, key="selected_model_stage2", on_change=on_model_change
        )
    with col3:
        selected_model3 = st.selectbox(
            "Choose a transformation & normalization model:", model_names3, key="selected_model_stage3", on_change=on_model_change
        )

    # Text input with change detection
    user_input = st.text_input(
        "Enter text for all in one inference:", key="user_input_stage3", on_change=on_text_change
    )

    if st.session_state.get("last_processed_input", "") != user_input:
        st.session_state.prediction_generated = False
        st.session_state.feedback_submitted = False

    st.markdown("#### Generation Parameters")
    col1, col2 = st.columns(2)

    with col1:
        use_beam = st.checkbox("Use Beam Search", value=False, on_change=on_beam_checkbox_change)
        if use_beam:
            beams = st.number_input("Number of beams:", min_value=1, max_value=10, value=3, step=1, on_change=on_beam_checkbox_change)
            do_sample = False
            temp = None
            top_p = None
            top_k = None
        else:
            beams = None
            do_sample = st.checkbox("Enable Sampling", value=True, on_change=on_enable_sampling_checkbox_change)
            temp = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=0.4, step=0.1, on_change=on_temperature_change) if do_sample else None

    with col2:
        top_p = st.slider("Top-p (nucleus sampling):", min_value=0.0, max_value=1.0, value=0.9, step=0.05, on_change=on_top_p_change) if (not use_beam and do_sample) else None
        model_config = MODEL_OPTIONS3[selected_model3]
        max_top_k = model_config.get("max_top_k", 50)
        if not use_beam and do_sample:
            col_slider, col_input = st.columns(2)
            st.write("Top-K: Top K most probable tokens, recommended range: 10-60")
            with col_slider:
                top_k_slider = st.slider(
                    "Top-k (slider):",
                    min_value=0,
                    max_value=max_top_k,
                    value=st.session_state.top_k,
                    step=1,
                    key="top_k_slider",
                    on_change=update_top_k_from_slider
                )
            with col_input:
                top_k_input = st.number_input(
                    "Top-k (number input):",
                    min_value=0,
                    max_value=max_top_k,
                    value=st.session_state.top_k,
                    step=1,
                    key="top_k_input",
                    on_change=update_top_k_from_input
                )
            final_top_k = st.session_state.top_k
        else:
            final_top_k = None

    col_tokens, col_return = st.columns(2)
    with col_tokens:
        max_new_tokens = st.number_input("Max New Tokens:", min_value=1, value=1024, step=1, on_change=on_max_new_tokens_change)
        early_stopping = st.checkbox("Early Stopping", value=True, on_change=on_enable_earlyStopping_checkbox_change)
    with col_return:
        if beams is not None:
            num_return_sequences = st.number_input(
                "Num Return Sequences:",
                min_value=1,
                max_value=beams,
                value=1,
                step=1,
                on_change=on_num_return_sequences_change
            )
        else:
            num_return_sequences = st.number_input(
                "Num Return Sequences:",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
                on_change=on_num_return_sequences_change
            )
        user_input_copy = user_input

    current_time = time.time()
    if user_input.strip() and (current_time - st.session_state.last_change >= 1.25) and st.session_state.get("prediction_generated", False) is False:
        st.session_state.last_processed_input = user_input
        
        progress_bar = st.progress(0)
        update_progress(progress_bar, 0, 10)
        col_spinner, col_warning = st.columns(2)

        with col_warning:
            warning_placeholder = st.empty()
            warning_placeholder.warning("Don't change the text data or any input parameters or switch models or pages while inference is loading...")

        with col_spinner:
            with st.spinner("Please wait, inference is loading..."):
                model1, tokenizer1, predict_func1 = load_selected_model1(selected_model1)
                model2, tokenizer2, predict_func2 = load_selected_model2(selected_model2)
                model3, tokenizer3, predict_func3 = load_selected_model3(selected_model3)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if model1 is None:
                    st.error("⚠️ Error: Model 1 failed to load!")
                    st.stop()
                if hasattr(model1, "to"):
                    model1.to(device)
                if model2 is None:
                    st.error("⚠️ Error: Model 2 failed to load!")
                    st.stop()
                if hasattr(model2, "to"):
                    model2.to(device)
                if model3 is None:
                    st.error("⚠️ Error: Model 3 failed to load!")
                    st.stop()
                if hasattr(model3, "to"):
                    model3.to(device)
                predictions1 = predict_func1(user_input, model1, tokenizer1, device)
                predictions2 = predict_func2(user_input, model2, tokenizer2, device)
                predictions = predict_func3(
                    model3, tokenizer3, user_input, device,
                    num_return_sequences,
                    beams,
                    do_sample,
                    temp,
                    top_p,
                    final_top_k,
                    max_new_tokens,
                    early_stopping
                )

        update_progress(progress_bar, 10, 100)

        warning_placeholder.empty()

        st.session_state.predictions = predictions
        st.session_state.predictions1 = predictions1
        st.session_state.predictions2 = predictions2
        print(predictions1)
        print(predictions2)
        if len(predictions) > 1:
            st.write("### Most Probable Predictions:")
            for i, pred in enumerate(predictions, start=1):
                st.markdown(f"**Prediction Sequence {i}:** {pred}")
        else:
            st.write("### Predicted Sequence:")
            st.write(predictions[0])

        graph_code = get_sentiment_emotion_graph_code(user_input, predictions[0], predictions1, predictions2)
        st.session_state.graphviz_code = graph_code
    
    # Now display the graph from session state:
        st.graphviz_chart(st.session_state.graphviz_code)
        progress_bar.empty()
    # else:
    #     st.info("Waiting for input to settle...")
    
        # Mark that a prediction has been generated
        st.session_state.prediction_generated = True
        
    else:
        # If predictions are already generated, display the stored ones
        if st.session_state.get("predictions") and st.session_state.get("graphviz_code") and st.session_state.get("predictions2") and st.session_state.get("predictions1"):
            predictions = st.session_state.predictions
            if len(predictions) > 1:
                st.write("### Most Probable Predictions:")
                for i, pred in enumerate(predictions, start=1):
                    st.markdown(f"**Prediction Sequence {i}:** {pred}")
            else:
                st.write("### Predicted Sequence:")
                st.write(predictions[0])
            st.graphviz_chart(st.session_state.graphviz_code)