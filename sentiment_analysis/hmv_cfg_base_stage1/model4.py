import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

from imports import *

import importlib.util
import os
import sys
import joblib

import torch
import torch.nn as nn
import torch.functional as F
from transformers import DebertaV2Model, DebertaV2Tokenizer, AutoModel, AutoTokenizer
import safetensors
# from safetensors import load_file, save_file
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, safe_open


import pickle
import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
# from keras_preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "config", "stage1_models.json")

MODEL_OPTIONS = {
    "4": {
        "name": "LSTM Custom Model",
        "type": "lstm_uncased_custom",
        "module_path": "hmv_cfg_base_stage1.model4",
        "hf_location": "tachygraphy-microtrext-norm-org/LSTM-LV1-SentimentPolarities",
        "tokenizer_class": "",
        "model_class": "",
        "problem_type": "multi_label_classification",
        "base_model": "",
        "base_model_class": "",
        "num_labels": 3,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}


model_key = "4"
model_info = MODEL_OPTIONS[model_key]
hf_location = model_info["hf_location"]


@st.cache_resource
def load_model():
    repo_id = hf_location
    print("Loading model 4")
    model_path = hf_hub_download(repo_id=repo_id, filename="lstm.h5")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pickle")

    lstm_model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Model 4 loaded")

    return lstm_model, tokenizer


def pad_sequences_custom(sequences, maxlen, dtype="int32", padding="pre", truncating="pre", value=0):
    """
    Pads each sequence to the same length (maxlen).

    Args:
        sequences (list of list of int): A list where each element is a sequence (list of integers).
        maxlen (int): Maximum length of all sequences.
        dtype (str): Data type of the output (default "int32").
        padding (str): 'pre' or 'post'—whether to add padding before or after the sequence.
        truncating (str): 'pre' or 'post'—whether to remove values from the beginning or end if sequence is too long.
        value (int): The padding value.

    Returns:
        numpy.ndarray: 2D array of shape (number of sequences, maxlen)
    """
    # Initialize a numpy array with the pad value.
    num_samples = len(sequences)
    padded = np.full((num_samples, maxlen), value, dtype=dtype)
    
    for i, seq in enumerate(sequences):
        if not seq:
            continue  # skip empty sequences
        if len(seq) > maxlen:
            if truncating == "pre":
                trunc = seq[-maxlen:]
            elif truncating == "post":
                trunc = seq[:maxlen]
            else:
                raise ValueError("Invalid truncating type: choose 'pre' or 'post'.")
        else:
            trunc = seq
        if padding == "post":
            padded[i, :len(trunc)] = trunc
        elif padding == "pre":
            padded[i, -len(trunc):] = trunc
        else:
            raise ValueError("Invalid padding type: choose 'pre' or 'post'.")
    
    return padded

def predict(text, model, tokenizer, device, max_len=128):
# def predict(text, model, tokenizer, max_len=128):
    # Convert text to a sequence of integers
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequence using our custom padding function
    padded_sequences = pad_sequences_custom(sequences, maxlen=max_len, dtype="int32", value=0)
    # Get the model's output (logits); assume shape is (1, num_classes)
    logits = model.predict(padded_sequences, batch_size=1, verbose=0)[0]
    print(logits)
    # Convert logits to probabilities using the exponential and normalize (softmax)
    # exp_logits = np.exp(logits)
    # probabilities = exp_logits / np.sum(exp_logits)
    
    # Ensure the output is a 2D array: shape (1, 3)
    probabilities = logits / logits.sum()
    print(probabilities)
    probabilities = np.atleast_2d(probabilities)
    print(probabilities)
    return probabilities


# def predict(text, model, tokenizer, max_len=128):
#     sequences = tokenizer.texts_to_sequences([text])
#     # Use our custom pad_sequences function:
#     padded_sequences = pad_sequences_custom(sequences, maxlen=max_len, dtype="int32", value=0)
#     prediction = model.predict(padded_sequences, batch_size=1, verbose=0)[0]
#     pred_class = np.argmax(prediction)
#     sentiment_labels = ["Negative", "Neutral", "Positive"]
#     probabilities = prediction / prediction.sum()
#     return sentiment_labels[pred_class], pred_class, probabilities


