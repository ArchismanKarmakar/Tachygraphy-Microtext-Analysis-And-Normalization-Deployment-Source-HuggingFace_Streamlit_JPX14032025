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

from dotenv import load_dotenv
# import psycopg2
from supabase import create_client, Client
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

env_path = os.path.join(os.path.dirname(__file__), "..", ".devcontainer", ".env")

# from transformers.utils import move_cache_to_trash
# from huggingface_hub import delete_cache


# from hmv_cfg_base_stage1.model1 import load_model as load_model1
# from hmv_cfg_base_stage1.model1 import predict as predict1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE3 = os.path.join(BASE_DIR, "config", "stage3_models.json")
LOADERS_STAGE3 = os.path.join(BASE_DIR, "hmv_cfg_base_stage3")


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
def load_model_config():
    with open(CONFIG_STAGE3, "r") as f:
        model_data = json.load(f)

    # Extract names for dropdown
    model_options = {v["name"]: v for v in model_data.values()}
    return model_data, model_options


MODEL_DATA, MODEL_OPTIONS = load_model_config()


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


def load_selected_model(model_name):
    global current_model, current_tokenizer

    # st.cache_resource.clear()

    # free_memory()

    # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
    # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[model_name])  # ✅ Check selected model
    # st.write("DEBUG: Model Name:", model_name)  # ✅ Check selected model

    if model_name not in MODEL_OPTIONS:
        st.error(f"⚠️ Model '{model_name}' not found in config!")
        return None, None, None

    model_info = MODEL_OPTIONS[model_name]
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
    st.session_state.last_processed_input = ""


# Function to update session state when text changes


def on_text_change():
    st.session_state.text_changed = True  # Mark text as changed

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""


def update_top_k_from_slider():
    st.session_state.top_k = st.session_state.top_k_slider

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""


def update_top_k_from_input():
    st.session_state.top_k = st.session_state.top_k_input

    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_temperature_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_top_p_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_beam_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_enable_sampling_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_enable_earlyStopping_checkbox_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_max_new_tokens_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

def on_num_return_sequences_change():
    st.session_state.prediction_generated = False
    st.session_state.feedback_submitted = False
    st.session_state.predictions = None
    # st.session_state.last_processed_input = ""

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(MODEL_OPTIONS.keys())[
        0]  # Default model
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


def transform_and_normalize():
    # No cache clearing here—only in the model change callback!

    # st.write(st.session_state)

    if "last_change" not in st.session_state:
        st.session_state.last_change = time.time()
    if "auto_predict_triggered" not in st.session_state:
        st.session_state.auto_predict_triggered = False

    
    if "top_k" not in st.session_state:
        st.session_state.top_k = 50

    model_names = list(MODEL_OPTIONS.keys())

    # Check if the stored selected model is valid; if not, reset it
    if "selected_model" in st.session_state:
        if st.session_state.selected_model not in model_names:
            st.session_state.selected_model = model_names[0]
    else:
        st.session_state.selected_model = model_names[0]

    st.title("Stage 3: Text Transformation & Normalization")
    st.write("This section handles the transformation and normalization of informal text into standard formal English.")

    # Model selection with change detection; clearing cache happens in on_model_change()
    selected_model = st.selectbox(
        "Choose a model:", model_names, key="selected_model_stage3", on_change=on_model_change
    )

    # Text input with change detection
    user_input = st.text_input(
        "Enter text for normalization:", key="user_input_stage3", on_change=on_text_change
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
        model_config = MODEL_OPTIONS[selected_model]
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
                model, tokenizer, predict_func = load_selected_model(selected_model)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if model is None:
                    st.error("⚠️ Error: Model failed to load!")
                    st.stop()
                if hasattr(model, "to"):
                    model.to(device)
                predictions = predict_func(
                    model, tokenizer, user_input, device,
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
        if len(predictions) > 1:
            st.write("### Most Probable Predictions:")
            for i, pred in enumerate(predictions, start=1):
                st.markdown(f"**Prediction Sequence {i}:** {pred}")
        else:
            st.write("### Predicted Sequence:")
            st.write(predictions[0])
        progress_bar.empty()
    # else:
    #     st.info("Waiting for input to settle...")
    
        # Mark that a prediction has been generated
        st.session_state.prediction_generated = True
        
    else:
        # If predictions are already generated, display the stored ones
        if st.session_state.get("predictions"):
            predictions = st.session_state.predictions
            if len(predictions) > 1:
                st.write("### Most Probable Predictions:")
                for i, pred in enumerate(predictions, start=1):
                    st.markdown(f"**Prediction Sequence {i}:** {pred}")
            else:
                st.write("### Predicted Sequence:")
                st.write(predictions[0])

    # Only show the feedback form if a prediction has been generated
    # if st.session_state.get("prediction_generated", False):
    #     if not st.session_state.get("feedback_submitted", False):
    #         with st.form("feedback_form", clear_on_submit=True, border=False):
    #             st.error("New API keys are coming in Q2 2025, May 1st, old API authentication will be deprecated and blocked by Postgrest.")
    #             st.warning("This form and database are running in test mode, please be careful with your data.")
    #             st.write("### Data Collection Form")
    #             st.write("#### If the predictions generated are wrong, please provide feedback to help improve the model.")
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 feedback = st.text_input(
    #                     "Enter the correct expanded standard formal English text:",
    #                     key="feedback_input"
    #                 )
    #             with col2:
    #                 feedback2 = st.text_input(
    #                     "Enter any one of the wrongly predicted text:",
    #                     key="feedback_input2"
    #                 )
    #             submit_feedback = st.form_submit_button("Submit Feedback")
    #             if submit_feedback and feedback.strip() and feedback2.strip():
    #                 data_to_insert = {
    #                     # "id" : str(uuid.uuid4()),            # text
    #                     # "created_at": datetime.now(timezone.utc).isoformat(),       # timestamp
    #                     "input_text": user_input,            # text
    #                     "correct_text_by_user": feedback,    # text
    #                     "model_used": selected_model,        # text
    #                     "wrong_pred_any": feedback2 if feedback2.strip() else ""
    #                 }
    #                 # Here we use the supabase client already created above
    #                 # supabase = get_connection()
    #                 # load_dotenv()
    #                 # print("SUPABASE_URL:", os.environ.get("SUPABASE_URL"))
    #                 # print("anon_key:", os.environ.get("anon_key"))
    #                 # print("table3_name:", os.environ.get("table3_name"))
    #                 # load_dotenv(dotenv_path=env_path)
    #                 # load_dotenv()
    #                 # supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("anon_key"))
    #                 # response = supabase.table(os.environ.get("table3_name")).insert(data_to_insert, returning="minimal").execute()
    #                 try:
    #                     supabase: Client = create_client(get_env_variable("SUPABASE_DB_TACHYGRAPHY_DB_URL"), get_env_variable("SUPABASE_DB_TACHYGRAPHY_ANON_API_KEY"))
    #                     response = supabase.table(get_env_variable("SUPABASE_DB_TACHYGRAPHY_DB_STAGE3_TABLE")).insert(data_to_insert, returning="minimal").execute()
    #                     st.success("Feedback submitted successfully!")
    #                     st.session_state.feedback_submitted = True
    #                 except Exception as e:
    #                     st.error(f"Feedback submission failed: {e}")

    #     else:
    #         st.info("Feedback already submitted for this prediction.")

if __name__ == "__main__":
    transform_and_normalize()









# # Main function to show the app
# def transform_and_normalize():

#     # st.cache_resource.clear()
#     # free_memory()

#     if "top_k" not in st.session_state:
#         st.session_state.top_k = 50

#     model_names = list(MODEL_OPTIONS.keys())

#     # Check if the stored selected model is valid; if not, reset it
#     if "selected_model" in st.session_state:
#         if st.session_state.selected_model not in model_names:
#             st.session_state.selected_model = model_names[0]
#     else:
#         st.session_state.selected_model = model_names[0]

#     st.title("Stage 3: Text Transformation & Normalization")
#     st.write("This section handles the transformation and normalization of informal text containing short-hands (microtexts), abbreviations, acronyms, slangs, multilingual conversational text etc. into readable, understandable standard formal English.")

#     # Model selection with change detection
#     selected_model = st.selectbox(
#         "Choose a model:", model_names, key="selected_model", on_change=on_model_change
#     )

#     # Text input with change detection
#     user_input = st.text_input(
#         "Enter text for emotions mood-tag analysis:", key="user_input", on_change=on_text_change
#     )

#     st.markdown("#### Generation Parameters")
#     col1, col2 = st.columns(2)

#     with col1:
#         use_beam = st.checkbox("Use Beam Search", value=False)
#         if use_beam:
#             beams = st.number_input(
#                 "Number of beams:", min_value=1, value=5, step=1)
#             do_sample = False
#             temp = None
#             top_p = None
#             top_k = None
#         else:
#             beams = None
#             do_sample = st.checkbox("Enable Sampling", value=True)
#             temp = st.slider("Temperature:", min_value=0.1, max_value=2.0,
#                              value=0.7, step=0.1) if do_sample else None

#     with col2:
#         top_p = st.slider("Top-p (nucleus sampling):", min_value=0.0, max_value=1.0,
#                           value=0.9, step=0.05) if not use_beam and do_sample else None
#         model_config = MODEL_OPTIONS[selected_model]
#         max_top_k = model_config.get("max_top_k", 50)
#         # top_k = st.number_input("Top-k:", min_value=0, value=50, step=1) if not use_beam and do_sample else None
#         # top_k = st.slider("Top-k:", min_value=0, max_value=max_top_k, value=50, step=1) if (not use_beam and do_sample) else None

#         if not use_beam and do_sample:

#             col_slider, col_input = st.columns(2)

#             with col_slider:
#                 top_k_slider = st.slider(
#                     "Top-k (slider):",
#                     min_value=0,
#                     max_value=max_top_k,
#                     value=st.session_state.top_k,
#                     step=1,
#                     key="top_k_slider",
#                     on_change=update_top_k_from_slider
#                 )
#             with col_input:
#                 top_k_input = st.number_input(
#                     "Top-k (number input):",
#                     min_value=0,
#                     max_value=max_top_k,
#                     value=st.session_state.top_k,
#                     step=1,
#                     key="top_k_input",
#                     on_change=update_top_k_from_input
#                 )
#             final_top_k = st.session_state.top_k
#         else:
#             final_top_k = None

#     # max_new_tokens = st.number_input("Max New Tokens:", min_value=1, value=1024, step=1)
#     # early_stopping = st.checkbox("Early Stopping", value=True)
#     # num_return_sequences = st.number_input("Num Return Sequences:", min_value=1, value=1, step=1)

#     col_tokens, col_return = st.columns(2)

#     with col_tokens:
#         max_new_tokens = st.number_input(
#             "Max New Tokens:", min_value=1, value=1024, step=1)
#         early_stopping = st.checkbox("Early Stopping", value=True)

#     with col_return:
#         if beams is not None:
#             num_return_sequences = st.number_input(
#                 "Num Return Sequences:",
#                 min_value=1,
#                 max_value=beams,
#                 value=1,
#                 step=1
#             )
#         else:
#             num_return_sequences = st.number_input(
#                 "Num Return Sequences:",
#                 min_value=1,
#                 max_value=3,
#                 value=1,
#                 step=1
#             )

#         user_input_copy = user_input



#     current_time = time.time()
#     if user_input.strip() and (current_time - st.session_state.last_change >= 1.5):
#         # Reset change flag (if needed)
#         st.session_state.last_processed_input = user_input
        
#         progress_bar = st.progress(0)
#         update_progress(progress_bar, 0, 10)
#         with st.spinner("Predicting..."):
#             model, tokenizer, predict_func = load_selected_model(selected_model)
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             if model is None:
#                 st.error("⚠️ Error: Model failed to load!")
#                 st.stop()
#             if hasattr(model, "to"):
#                 model.to(device)
#             predictions = predict_func(
#                 model, tokenizer, user_input, device,
#                 num_return_sequences,
#                 beams,
#                 do_sample,
#                 temp,
#                 top_p,
#                 final_top_k,
#                 max_new_tokens,
#                 early_stopping
#             )
#         update_progress(progress_bar, 10, 100)
        
#         if len(predictions) > 1:
#             st.write("### Multiple Predictions:")
#             for i, pred in enumerate(predictions, start=1):
#                 st.markdown(f"**Sequence {i}:** {pred}")
#         else:
#             st.write("### Prediction:")
#             st.write(predictions[0])
#         progress_bar.empty()
#     else:
#         st.info("Waiting for input to settle...")

    # Only run inference if:
    # 1. The text is NOT empty
    # 2. The text has changed OR the model has changed
    # auto_predict = False
    # if user_input.strip():
    #     if (user_input != st.session_state.last_processed_input) or st.session_state.model_changed:
    #         auto_predict = True

    # if auto_predict:
    #     run_inference = True
    # else:
    #     run_inference = st.button("Run Prediction")

    # if run_inference and user_input.strip():
    #     # Reset change flags and update last processed input
    #     st.session_state.last_processed_input = user_input
    #     st.session_state.model_changed = False
    #     st.session_state.text_changed = False

    #     progress_bar = st.progress(0)
    #     update_progress(progress_bar, 0, 10)

    #     with st.spinner("Please wait..."):
    #         model, tokenizer, predict_func = load_selected_model(selected_model)
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         if model is None:
    #             st.error("⚠️ Error: Model failed to load! Check model selection or configuration.")
    #             st.stop()
    #         if hasattr(model, "to"):
    #             model.to(device)

    #         predictions = predict_func(
    #             model, tokenizer, user_input, device,
    #             num_return_sequences,
    #             beams,
    #             do_sample,
    #             temp,
    #             top_p,
    #             final_top_k,
    #             max_new_tokens,
    #             early_stopping
    #         )
    #     update_progress(progress_bar, 10, 100)

    #     if len(predictions) > 1:
    #         st.write("### Multiple Predicted Transformed & Normalized Texts:")
    #         for i, pred in enumerate(predictions, start=1):
    #             st.markdown(f"**Sequence {i}:** {pred}")
    #     else:
    #         st.write("### Predicted Transformed & Normalized Text:")
    #         st.write(predictions[0])
    #     progress_bar.empty()


# if __name__ == "__main__":
#     # st.cache_resource.clear()
#     # free_memory()
#     transform_and_normalize()
#     # show_dashboard()
#     # show_emotion_analysis()
#     # show_sentiment_analysis()
#     # show_text_transformation()
