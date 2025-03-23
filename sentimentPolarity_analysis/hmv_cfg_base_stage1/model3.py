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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "config", "stage1_models.json")

MODEL_OPTIONS = {
    "3": {
        "name": "BERT Base Uncased Custom Model",
        "type": "bert_base_uncased_custom",
        "module_path": "hmv_cfg_base_stage1.model3",
        "hf_location": "https://huggingface.co/tachygraphy-microtext-normalization-iemk/BERT-LV1-SentimentPolarities/resolve/main/saved_weights.pt",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "BERT_architecture",
        "problem_type": "multi_label_classification",
        "base_model": "bert-base-uncased",
        "base_model_class": "AutoModel",
        "num_labels": 3,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}

class BERT_architecture(nn.Module):

    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert
        
        self.dropout = nn.Dropout(0.3)  # Increased dropout for regularization
        self.layer_norm = nn.LayerNorm(768)  # Layer normalization
        
        self.fc1 = nn.Linear(768, 256)  # Dense layer
        self.fc2 = nn.Linear(256, 3)  # Output layer with 3 classes
        
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, token_type_ids):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        x = self.layer_norm(cls_hs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    

model_key = "3"
model_info = MODEL_OPTIONS[model_key]
hf_location = model_info["hf_location"]
base_model = model_info["base_model"]
base_model_class = globals()[model_info["base_model_class"]]

tokenizer_class = globals()[model_info["tokenizer_class"]]
model_class = globals()[model_info["model_class"]]
    

@st.cache_resource
def load_model():
    bert = base_model_class.from_pretrained(base_model)
    tokenizer = tokenizer_class.from_pretrained(base_model)
    print("Loading model 3")

    model = BERT_architecture(bert)
    state_dict = torch.hub.load_state_dict_from_url(
        hf_location,
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(state_dict)
    print("Model 3 loaded")
    
    return model, tokenizer


def predict(input_text, model, tokenizer, device, max_seq_len=128):
    inputs = tokenizer(
        input_text,
        add_special_tokens=True,
        padding=True,
        truncation=True,  # Ensure dynamic length truncation
        max_length=max_seq_len,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt',
    ).to(device)


    with torch.no_grad():
        # outputs = model(**inputs)

        outputs = model(
            sent_id=inputs["input_ids"],             # input_ids → sent_id
            mask=inputs["attention_mask"],           # attention_mask → mask
            token_type_ids=inputs["token_type_ids"]  # token_type_ids → token_type_ids
        )
    #     preds = outputs.cpu().numpy()
    #     pred = np.argmax(preds, axis=1)
    
    # return pred

    # Ensure output shape consistency
    # if outputs.dim() == 1:
    #     # Reshape to [1, num_classes] if it's a single prediction
    #     outputs = outputs.unsqueeze(0)

    # Apply softmax here if you need probabilities
    # probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    ## TO RETURN ARGMAX VALUE
    # pred_class = torch.argmax(outputs, dim=1).cpu().numpy()

    # num_classes=3

    # probabilities = np.zeros((1, num_classes))
    # probabilities[0, pred_class] = 1.0

    ## TO RETURN PROBABILITIES FROM LOG SOFTMAX OF MODEL
    probabilities = torch.exp(outputs).cpu().numpy()

    print(probabilities)
    
    return probabilities


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully.")