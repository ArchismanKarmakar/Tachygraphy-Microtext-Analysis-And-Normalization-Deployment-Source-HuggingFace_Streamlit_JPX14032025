import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

from imports import *
import importlib.util
import os
import sys
import joblib
import time
import torch
# from transformers.utils import move_cache_to_trash 
# from huggingface_hub import delete_cache
from transformers.utils.hub import TRANSFORMERS_CACHE
import shutil


# from hmv_cfg_base_stage1.model1 import load_model as load_model1
# from hmv_cfg_base_stage1.model1 import predict as predict1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "config", "stage1_models.json")
LOADERS_STAGE1 = os.path.join(BASE_DIR, "hmv-cfg-base-stage1")


SENTIMENT_POLARITY_LABELS = [
    "negative", "neutral", "positive"
]

current_model = None
current_tokenizer = None

# Enabling Resource caching


# @st.cache_resource
def load_model_config():
    with open(CONFIG_STAGE1, "r") as f:
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
def update_progress(progress_bar, start, end, delay=0.1):
    for i in range(start, end + 1, 5):  # Increment in steps of 5%
        progress_bar.progress(i)
        time.sleep(delay)  # Simulate processing time
        # st.experimental_rerun() # Refresh the page


# Function to update session state when model changes
def on_model_change():
    st.session_state.model_changed = True  # Mark model as changed
    

# Function to update session state when text changes


def on_text_change():
    st.session_state.text_changed = True  # Mark text as changed


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


def show_sentiment_analysis():

    model_names = list(MODEL_OPTIONS.keys())

    # Check if the stored selected model is valid; if not, reset it
    if "selected_model" in st.session_state:
        if st.session_state.selected_model not in model_names:
            st.session_state.selected_model = model_names[0]
    else:
        st.session_state.selected_model = model_names[0]

    st.title("Stage 1: Sentiment Polarity Analysis")
    st.write("This section handles sentiment analysis.")

    # Model selection with change detection
    selected_model = st.selectbox(
        "Choose a model:", list(MODEL_OPTIONS.keys()), key="selected_model", on_change=on_model_change
    )

    # Text input with change detection
    user_input = st.text_input(
        "Enter text for sentiment analysis:", key="user_input", on_change=on_text_change
    )
    user_input_copy = user_input

    # Only run inference if:
    # 1. The text is NOT empty
    # 2. The text has changed OR the model has changed
    if user_input.strip() and (st.session_state.text_changed or st.session_state.model_changed):

        # disable_ui()


        # Reset session state flags
        st.session_state.last_processed_input = user_input
        st.session_state.model_changed = False
        st.session_state.text_changed = False   # Store selected model

        # ADD A DYNAMIC PROGRESS BAR
        progress_bar = st.progress(0)
        update_progress(progress_bar, 0, 10)
        # status_text = st.empty()

        # update_progress(0, 10)
        # status_text.text("Loading model...")

        # Make prediction

        # model, tokenizer = load_model()
        # model, tokenizer = load_selected_model(selected_model)
        with st.spinner("Please wait..."):
            model, tokenizer, predict_func = load_selected_model(selected_model)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if model is None:
                st.error(
                    "⚠️ Error: Model failed to load! Check model selection or configuration.")
                st.stop()

            # model.to(device)
            if hasattr(model, "to"):
                model.to(device)

            # predictions = predict(user_input, model, tokenizer, device)

            predictions = predict_func(user_input, model, tokenizer, device)

            # Squeeze predictions to remove extra dimensions
            predictions_array = predictions.squeeze()

            # Convert to binary predictions (argmax)
            binary_predictions = np.zeros_like(predictions_array)
            max_indices = np.argmax(predictions_array)
            binary_predictions[max_indices] = 1

            # Update progress bar for prediction and model loading
            update_progress(progress_bar, 10, 100)

        # Display raw predictions
        st.write(f"**Predicted Sentiment Scores:** {predictions_array}")

        # enable_ui()
##
        # Display binary classification result
        st.write(f"**Predicted Sentiment:**")
        st.write(f"**NEGATIVE:** {binary_predictions[0]}, **NEUTRAL:** {binary_predictions[1]}, **POSITIVE:** {binary_predictions[2]}")
        # st.write(f"**NEUTRAL:** {binary_predictions[1]}")
        # st.write(f"**POSITIVE:** {binary_predictions[2]}")

        # 1️⃣ **Polar Plot (Plotly)**
        sentiment_polarities = predictions_array.tolist()
        fig_polar = px.line_polar(
            pd.DataFrame(dict(r=sentiment_polarities,
                         theta=SENTIMENT_POLARITY_LABELS)),
            r='r', theta='theta', line_close=True
        )
        st.plotly_chart(fig_polar)

        # 2️⃣ **Normalized Horizontal Bar Chart (Matplotlib)**
        normalized_predictions = predictions_array / predictions_array.sum()

        fig, ax = plt.subplots(figsize=(8, 2))
        left = 0
        for i in range(len(normalized_predictions)):
            ax.barh(0, normalized_predictions[i], color=plt.cm.tab10(
                i), left=left, label=SENTIMENT_POLARITY_LABELS[i])
            left += normalized_predictions[i]

        # Configure the chart
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.legend(loc='upper center', bbox_to_anchor=(
            0.5, -0.15), ncol=len(SENTIMENT_POLARITY_LABELS))
        plt.title("Sentiment Polarity Prediction Distribution")

        # Display in Streamlit
        st.pyplot(fig)

        progress_bar.empty()


if __name__ == "__main__":
    show_sentiment_analysis()










#########


# def show_sentiment_analysis():
#     st.cache_resource.clear()
#     free_memory()

#     st.title("Stage 1: Sentiment Polarity Analysis")
#     st.write("This section handles sentiment analysis.")

#     # Model selection with change detection
#     selected_model = st.selectbox(
#         "Choose a model:", list(MODEL_OPTIONS.keys()), key="selected_model", on_change=on_model_change
#     )

#     # Text input with change detection
#     user_input = st.text_input(
#         "Enter text for sentiment analysis:", key="user_input", on_change=on_text_change
#     )
#     user_input_copy = user_input

#     # Only run inference if:
#     # 1. The text is NOT empty
#     # 2. The text has changed OR the model has changed
#     if user_input.strip() and (st.session_state.text_changed or st.session_state.model_changed):

#         # Reset session state flags
#         st.session_state.last_processed_input = user_input
#         st.session_state.model_changed = False
#         st.session_state.text_changed = False   # Store selected model

#         # ADD A DYNAMIC PROGRESS BAR
#         progress_bar = st.progress(0)
#         update_progress(progress_bar, 0, 10)
#         # status_text = st.empty()

#         # update_progress(0, 10)
#         # status_text.text("Loading model...")

#         # Make prediction

#         # model, tokenizer = load_model()
#         # model, tokenizer = load_selected_model(selected_model)
#         with st.spinner("Please wait..."):
#             model, tokenizer, predict_func = load_selected_model(selected_model)
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#             if model is None:
#                 st.error(
#                     "⚠️ Error: Model failed to load! Check model selection or configuration.")
#                 st.stop()

#             model.to(device)

#             # predictions = predict(user_input, model, tokenizer, device)

#             predictions = predict_func(user_input, model, tokenizer, device)

#             # Squeeze predictions to remove extra dimensions
#             predictions_array = predictions.squeeze()

#             # Convert to binary predictions (argmax)
#             binary_predictions = np.zeros_like(predictions_array)
#             max_indices = np.argmax(predictions_array)
#             binary_predictions[max_indices] = 1

#             # Update progress bar for prediction and model loading
#             update_progress(progress_bar, 10, 100)

#         # Display raw predictions
#         st.write(f"**Predicted Sentiment Scores:** {predictions_array}")

#         # Display binary classification result
#         st.write(f"**Predicted Sentiment:**")
#         st.write(
#             f"**NEGATIVE:** {binary_predictions[0]}, **NEUTRAL:** {binary_predictions[1]}, **POSITIVE:** {binary_predictions[2]}")
#         # st.write(f"**NEUTRAL:** {binary_predictions[1]}")
#         # st.write(f"**POSITIVE:** {binary_predictions[2]}")

#         # 1️⃣ **Polar Plot (Plotly)**
#         sentiment_polarities = predictions_array.tolist()
#         fig_polar = px.line_polar(
#             pd.DataFrame(dict(r=sentiment_polarities,
#                          theta=SENTIMENT_POLARITY_LABELS)),
#             r='r', theta='theta', line_close=True
#         )
#         st.plotly_chart(fig_polar)

#         # 2️⃣ **Normalized Horizontal Bar Chart (Matplotlib)**
#         normalized_predictions = predictions_array / predictions_array.sum()

#         fig, ax = plt.subplots(figsize=(8, 2))
#         left = 0
#         for i in range(len(normalized_predictions)):
#             ax.barh(0, normalized_predictions[i], color=plt.cm.tab10(
#                 i), left=left, label=SENTIMENT_POLARITY_LABELS[i])
#             left += normalized_predictions[i]

#         # Configure the chart
#         ax.set_xlim(0, 1)
#         ax.set_yticks([])
#         ax.set_xticks(np.arange(0, 1.1, 0.1))
#         ax.legend(loc='upper center', bbox_to_anchor=(
#             0.5, -0.15), ncol=len(SENTIMENT_POLARITY_LABELS))
#         plt.title("Sentiment Polarity Prediction Distribution")

#         # Display in Streamlit
#         st.pyplot(fig)

#         progress_bar.empty()

######
########



# def show_sentiment_analysis():
#     st.cache_resource.clear()
#     free_memory()

#     st.title("Stage 1: Sentiment Polarity Analysis")
#     st.write("This section handles sentiment analysis.")

#     # Model selection with change detection
#     selected_model = st.selectbox(
#         "Choose a model:", list(MODEL_OPTIONS.keys()), key="selected_model", on_change=on_model_change, disabled=st.session_state.disabled
#     )

#     # Text input with change detection
#     user_input = st.text_input(
#         "Enter text for sentiment analysis:", key="user_input", on_change=on_text_change, disabled=st.session_state.disabled
#     )
#     user_input_copy = user_input

#     # progress_bar = st.progress(0)
#     progress_bar = st.empty()

#     if st.session_state.disabled is False and st.session_state.predictions is not None:
#         st.write(f"**Predicted Sentiment Scores:** {st.session_state.predictions}")
#         st.write(f"**NEGATIVE:** {st.session_state.binary_predictions[0]}, **NEUTRAL:** {st.session_state.binary_predictions[1]}, **POSITIVE:** {st.session_state.binary_predictions[2]}")
#         st.plotly_chart(st.session_state.polar_plot)
#         st.pyplot(st.session_state.bar_chart)

#         update_progress(progress_bar, 95, 100)

#         st.session_state.predictions = None
#         st.session_state.binary_predictions = None
#         st.session_state.polar_plot = None
#         st.session_state.bar_chart = None
        
#         st.session_state.disabled = False
            
#         progress_bar.empty()


#     if user_input.strip() and (st.session_state.text_changed or st.session_state.model_changed) and st.session_state.disabled is False:
#         st.session_state.disabled = True
#         st.rerun()
#         return
        
    
#     if user_input.strip() and (st.session_state.text_changed or st.session_state.model_changed) and st.session_state.disabled is True:
#         # Mark processing as True to
        

#         # Reset session state flags
#         st.session_state.last_processed_input = user_input
#         st.session_state.model_changed = False
#         st.session_state.text_changed = False   # Store selected model

#         # ADD A DYNAMIC PROGRESS BAR
#         progress_bar = st.progress(0)
#         update_progress(progress_bar, 0, 10)
#         # status_text = st.empty()

#         # update_progress(0, 10)
#         # status_text.text("Loading model...")

#         # Make prediction

#         # model, tokenizer = load_model()
#         # model, tokenizer = load_selected_model(selected_model)
#         with st.spinner("Please wait..."):
#             model, tokenizer, predict_func = load_selected_model(selected_model)
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#             if model is None:
#                 st.error(
#                     "⚠️ Error: Model failed to load! Check model selection or configuration.")
#                 st.session_state.disabled = False
#                 st.rerun()
#                 st.stop()
#                 return

#             model.to(device)

#             # predictions = predict(user_input, model, tokenizer, device)

#             predictions = predict_func(user_input, model, tokenizer, device)

#             # Squeeze predictions to remove extra dimensions
#             predictions_array = predictions.squeeze()

#             # Convert to binary predictions (argmax)
#             binary_predictions = np.zeros_like(predictions_array)
#             max_indices = np.argmax(predictions_array)
#             binary_predictions[max_indices] = 1

#             # Update progress bar for prediction and model loading
#             update_progress(progress_bar, 10, 75)

#         # Display raw predictions
#         # st.write(f"**Predicted Sentiment Scores:** {predictions_array}")
#         st.session_state.predictions = predictions_array

#         # Display binary classification result
#         # st.write(f"**Predicted Sentiment:**")
#         # st.write(f"**NEGATIVE:** {binary_predictions[0]}, **NEUTRAL:** {binary_predictions[1]}, **POSITIVE:** {binary_predictions[2]}")
#         st.session_state.binary_predictions = binary_predictions


#         # 1️⃣ **Polar Plot (Plotly)**
#         sentiment_polarities = predictions_array.tolist()
#         fig_polar = px.line_polar(
#             pd.DataFrame(dict(r=sentiment_polarities,
#                         theta=SENTIMENT_POLARITY_LABELS)),
#             r='r', theta='theta', line_close=True
#         )
#         # st.plotly_chart(fig_polar)
#         st.session_state.polar_plot = fig_polar

#         # 2️⃣ **Normalized Horizontal Bar Chart (Matplotlib)**
#         normalized_predictions = predictions_array / predictions_array.sum()

#         fig, ax = plt.subplots(figsize=(8, 2))
#         left = 0
#         for i in range(len(normalized_predictions)):
#             ax.barh(0, normalized_predictions[i], color=plt.cm.tab10(
#                 i), left=left, label=SENTIMENT_POLARITY_LABELS[i])
#             left += normalized_predictions[i]

#         # Configure the chart
#         ax.set_xlim(0, 1)
#         ax.set_yticks([])
#         ax.set_xticks(np.arange(0, 1.1, 0.1))
#         ax.legend(loc='upper center', bbox_to_anchor=(
#             0.5, -0.15), ncol=len(SENTIMENT_POLARITY_LABELS))
#         # plt.title("Sentiment Polarity Prediction Distribution")
#         # st.pyplot(fig)
#         st.session_state.bar_chart = fig
#         update_progress(progress_bar, 75, 95)

#         # progress_bar.empty()

#         if st.session_state.disabled is True:
#             st.session_state.disabled = False
#             st.rerun()
#             return
#         else:
#             return
        



#####


### COMMENTED OUT CODE ###


# def load_selected_model(model_name):
#     # """Load model and tokenizer based on user selection."""
#     global current_model, current_tokenizer

#     # Free memory before loading a new model
#     free_memory()

#     if model_name not in MODEL_OPTIONS:
#         st.error(f"⚠️ Model '{model_name}' not found in config!")
#         return None, None

#     model_info = MODEL_OPTIONS[model_name]
#     hf_location = model_info["hf_location"]

#     model_module = model_info["module_path"]
#     # load_function = "load_model"
#     # predict_function = "predict"

#     load_function = model_info["load_function"]
#     predict_function = model_info["predict_function"]

#     # tokenizer_class = globals()[model_info["tokenizer_class"]]
#     # model_class = globals()[model_info["model_class"]]

#     # tokenizer = tokenizer_class.from_pretrained(hf_location)


#     load_model_func = import_from_module(model_module, load_function)
#     predict_func = import_from_module(model_module, predict_function)

#     # # Load model
#     # if model_info["type"] == "custom_checkpoint" or model_info["type"] == "custom_model":
#     #     model = torch.load(hf_location, map_location="cpu")  # Load PyTorch model
#     # elif model_info["type"] == "hf_automodel_finetuned_dbt3":
#     #     tokenizer_class = globals()[model_info["tokenizer_class"]]
#     #     model_class = globals()[model_info["model_class"]]
#     #     tokenizer = tokenizer_class.from_pretrained(hf_location)
#     #     model = model_class.from_pretrained(hf_location,
#     #                                         problem_type=model_info["problem_type"],
#     #                                         num_labels=model_info["num_labels"]
#     #     )
#     # else:
#     #     st.error("Invalid model selection")
#     #     return None, None


#     if load_model_func is None or predict_func is None:
#         st.error("❌ Model functions could not be loaded!")
#         return None, None

#     # current_model, current_tokenizer = model, tokenizer  # Store references
#     # return model, tokenizer

#     model, tokenizer = load_model_func(hf_location)

#     current_model, current_tokenizer = model, tokenizer
#     return model, tokenizer, predict_func


# def predict(text, model, tokenizer, device, max_len=128):
#     # Tokenize and pad the input text
#     inputs = tokenizer(
#         text,
#         add_special_tokens=True,
#         padding=True,
#         truncation=False,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)  # Move input tensors to the correct device

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Apply sigmoid activation (for BCEWithLogitsLoss)
#     probabilities = outputs.logits.cpu().numpy()

#     return probabilities

# def show_sentiment_analysis():

    # Add your sentiment analysis code here

    # user_input = st.text_input("Enter text for sentiment analysis:")
    # user_input = st.text_area("Enter text for sentiment analysis:", height=200)
    # user_input = st.text_area("Enter text for sentiment analysis:", max_chars=500)

# def show_sentiment_analysis():
#     st.title("Stage 1: Sentiment Polarity Analysis")
#     st.write("This section will handle sentiment analysis.")

#     if "selected_model" not in st.session_state:
#         st.session_state.selected_model = list(MODEL_OPTIONS.keys())[0]  # Default selection

#     if "clear_output" not in st.session_state:
#         st.session_state.clear_output = False

#     st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()), key="selected_model")

#     selected_model = st.session_state.selected_model

#     if selected_model not in MODEL_OPTIONS:
#         st.error(f"❌ Selected model '{selected_model}' not found!")
#         st.stop()

#     st.session_state.clear_output = True  # Reset output when model changes


#     # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
#     # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[selected_model])  # ✅ Check selected model


#     user_input = st.text_input("Enter text for sentiment analysis:")
#     user_input_copy = user_input

#     # if st.button("Run Analysis"):
#     #     if not user_input.strip():
#     #         st.warning("⚠️ Please enter some text before running analysis.")
#     #         return

#     # with st.form(key="sentiment_form"):
#     #     user_input = st.text_input("Enter text for sentiment analysis:")
#     #     submit_button = st.form_submit_button("Run Analysis")

#     #     user_input_copy = user_input

#     if user_input.strip():

#         ADD A DYNAMIC PROGRESS BAR
#         progress_bar = st.progress(0)
#         update_progress(progress_bar, 0, 10)
#         # status_text = st.empty()

#         # update_progress(0, 10)
#         # status_text.text("Loading model...")

#         # Make prediction

#         # model, tokenizer = load_model()
#         # model, tokenizer = load_selected_model(selected_model)

#         model, tokenizer, predict_func = load_selected_model(selected_model)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         if model is None:
#             st.error("⚠️ Error: Model failed to load! Check model selection or configuration.")
#             st.stop()

#         model.to(device)

#         # predictions = predict(user_input, model, tokenizer, device)

#         predictions = predict_func(user_input, model, tokenizer, device)

#         # Squeeze predictions to remove extra dimensions
#         predictions_array = predictions.squeeze()

#         # Convert to binary predictions (argmax)
#         binary_predictions = np.zeros_like(predictions_array)
#         max_indices = np.argmax(predictions_array)
#         binary_predictions[max_indices] = 1


#         # Update progress bar for prediction and model loading
#         update_progress(progress_bar, 10, 100)

#         # Display raw predictions
#         st.write(f"**Predicted Sentiment Scores:** {predictions_array}")

#         # Display binary classification result
#         st.write(f"**Predicted Sentiment:**")
#         st.write(f"**NEGATIVE:** {binary_predictions[0]}, **NEUTRAL:** {binary_predictions[1]}, **POSITIVE:** {binary_predictions[2]}")
#         # st.write(f"**NEUTRAL:** {binary_predictions[1]}")
#         # st.write(f"**POSITIVE:** {binary_predictions[2]}")

#         # 1️⃣ **Polar Plot (Plotly)**
#         sentiment_polarities = predictions_array.tolist()
#         fig_polar = px.line_polar(
#             pd.DataFrame(dict(r=sentiment_polarities, theta=SENTIMENT_POLARITY_LABELS)),
#             r='r', theta='theta', line_close=True
#         )
#         st.plotly_chart(fig_polar)

#         # 2️⃣ **Normalized Horizontal Bar Chart (Matplotlib)**
#         normalized_predictions = predictions_array / predictions_array.sum()

#         fig, ax = plt.subplots(figsize=(8, 2))
#         left = 0
#         for i in range(len(normalized_predictions)):
#             ax.barh(0, normalized_predictions[i], color=plt.cm.tab10(i), left=left, label=SENTIMENT_POLARITY_LABELS[i])
#             left += normalized_predictions[i]

#         # Configure the chart
#         ax.set_xlim(0, 1)
#         ax.set_yticks([])
#         ax.set_xticks(np.arange(0, 1.1, 0.1))
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(SENTIMENT_POLARITY_LABELS))
#         plt.title("Sentiment Polarity Prediction Distribution")

#         # Display in Streamlit
#         st.pyplot(fig)

#         progress_bar.empty()
