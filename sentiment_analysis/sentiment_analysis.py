from imports import *
import importlib.util
import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

# from hmv_cfg_base_stage1.model1 import load_model as load_model1
# from hmv_cfg_base_stage1.model1 import predict as predict1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "config", "stage1_models.json")
LOADERS_STAGE1 = os.path.join(BASE_DIR, "hmv-cfg-base-stage1")

# Load the model and tokenizer
# model_name = "tachygraphy-microtrext-norm-org/DeBERTa-v3-seqClassfication-LV1-SentimentPolarities-Batch8"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

SENTIMENT_POLARITY_LABELS = [
    "negative", "neutral", "positive"
]

current_model = None
current_tokenizer = None

# Enabling Resource caching
@st.cache_resource

def load_model_config():
    with open(CONFIG_STAGE1, "r") as f:
        model_data = json.load(f)

    model_options = {v["name"]: v for v in model_data.values()}  # Extract names for dropdown
    return model_data, model_options

MODEL_DATA, MODEL_OPTIONS = load_model_config()



# def load_model():
#     model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
#     tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
#     return model, tokenizer


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


def load_selected_model(model_name):
    global current_model, current_tokenizer

    free_memory()

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



def predict(text, model, tokenizer, device, max_len=128):
    # Tokenize and pad the input text
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)  # Move input tensors to the correct device

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid activation (for BCEWithLogitsLoss)
    probabilities = outputs.logits.cpu().numpy()

    return probabilities 

# def show_sentiment_analysis():

    # Add your sentiment analysis code here

    # user_input = st.text_input("Enter text for sentiment analysis:")
    # user_input = st.text_area("Enter text for sentiment analysis:", height=200)
    # user_input = st.text_area("Enter text for sentiment analysis:", max_chars=500)

def show_sentiment_analysis():
    st.title("Stage 1: Sentiment Polarity Analysis")
    st.write("This section will handle sentiment analysis.")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(MODEL_OPTIONS.keys())[0]  # Default selection

    if "clear_output" not in st.session_state:
        st.session_state.clear_output = False

    st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()), key="selected_model")

    selected_model = st.session_state.selected_model

    if selected_model not in MODEL_OPTIONS:
        st.error(f"❌ Selected model '{selected_model}' not found!")
        st.stop()

    st.session_state.clear_output = True  # Reset output when model changes


    # st.write("DEBUG: Available Models:", MODEL_OPTIONS.keys())  # ✅ See available models
    # st.write("DEBUG: Selected Model:", MODEL_OPTIONS[selected_model])  # ✅ Check selected model


    user_input = st.text_input("Enter text for sentiment analysis:")

    if user_input:
        # Make prediction

        # model, tokenizer = load_model()
        # model, tokenizer = load_selected_model(selected_model)

        model, tokenizer, predict_func = load_selected_model(selected_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None:
            st.error("⚠️ Error: Model failed to load! Check model selection or configuration.")
            st.stop()

        model.to(device)

        # predictions = predict(user_input, model, tokenizer, device)

        predictions = predict_func(user_input, model, tokenizer, device)

        # Squeeze predictions to remove extra dimensions
        predictions_array = predictions.squeeze()

        # Convert to binary predictions (argmax)
        binary_predictions = np.zeros_like(predictions_array)
        max_indices = np.argmax(predictions_array)
        binary_predictions[max_indices] = 1

        # Display raw predictions
        st.write(f"**Predicted Sentiment Scores:** {predictions_array}")

        # Display binary classification result
        st.write(f"**Predicted Sentiment:**")
        st.write(f"**NEGATIVE:** {binary_predictions[0]}, **NEUTRAL:** {binary_predictions[1]}, **POSITIVE:** {binary_predictions[2]}")
        # st.write(f"**NEUTRAL:** {binary_predictions[1]}")
        # st.write(f"**POSITIVE:** {binary_predictions[2]}")

        # 1️⃣ **Polar Plot (Plotly)**
        sentiment_polarities = predictions_array.tolist()
        fig_polar = px.line_polar(
            pd.DataFrame(dict(r=sentiment_polarities, theta=SENTIMENT_POLARITY_LABELS)), 
            r='r', theta='theta', line_close=True
        )
        st.plotly_chart(fig_polar)

        # 2️⃣ **Normalized Horizontal Bar Chart (Matplotlib)**
        normalized_predictions = predictions_array / predictions_array.sum()

        fig, ax = plt.subplots(figsize=(8, 2))
        left = 0
        for i in range(len(normalized_predictions)):
            ax.barh(0, normalized_predictions[i], color=plt.cm.tab10(i), left=left, label=SENTIMENT_POLARITY_LABELS[i])
            left += normalized_predictions[i]

        # Configure the chart
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(SENTIMENT_POLARITY_LABELS))
        plt.title("Sentiment Polarity Prediction Distribution")

        # Display in Streamlit
        st.pyplot(fig)
        


if __name__ == "__main__":
    show_sentiment_analysis()