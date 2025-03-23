import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "config", "stage1_models.json")

import torch
import torch.nn as nn
from imports import *
import torch.nn.functional as F




MODEL_OPTIONS = {
"1": {
        "name": "DeBERTa v3 Base for Sequence Classification",
        "type": "hf_automodel_finetuned_dbt3",
        "module_path": "hmv_cfg_base_stage1.model1",
        "hf_location": "tachygraphy-microtext-normalization-iemk/DeBERTa-v3-seqClassfication-LV1-SentimentPolarities-Batch8",
        "tokenizer_class": "DebertaV2Tokenizer",
        "model_class": "DebertaV2ForSequenceClassification",
        "problem_type": "multi_label_classification",
        "base_model": "microsoft/deberta-v3-base",
        "base_model_class": "DebertaV2ForSequenceClassification",
        "num_labels": 3,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}


model_key = "1"
model_info = MODEL_OPTIONS[model_key]
hf_location = model_info["hf_location"]

tokenizer_class = globals()[model_info["tokenizer_class"]]
model_class = globals()[model_info["model_class"]]


@st.cache_resource
def load_model():
    tokenizer = tokenizer_class.from_pretrained(hf_location)
    print("Loading model 1")
    model = model_class.from_pretrained(hf_location,
                                        problem_type=model_info["problem_type"],
                                        num_labels=model_info["num_labels"]
                                        )
    print("Model 1 loaded")

    return model, tokenizer


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

    # probabilities = outputs.logits.cpu().numpy()

    # probabilities = torch.relu(outputs.logits)
    # probabilities = torch.clamp(torch.tensor(probabilities), min=0.00000, max=1.00000).cpu().numpy()
    # probabilities /= probabilities.sum()
    # probabilities = probabilities.cpu().numpy()

    predictions = torch.sigmoid(outputs.logits).cpu().numpy()

    return predictions


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully.")
