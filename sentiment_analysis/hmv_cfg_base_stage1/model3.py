from imports import *

import importlib.util
import os
import sys
import joblib

import torch
import torch.nn as nn
import torch.functional as F
from transformers import DebertaV2Model, DebertaV2Tokenizer
import safetensors
# from safetensors import load_file, save_file
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, safe_open

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "config", "stage1_models.json")

MODEL_OPTIONS = {
    "3": {
        "name": "BERT Base Uncased Custom Model",
        "type": "db3_base_custom",
        "module_path": "hmv_cfg_base_stage1.model2",
        "hf_location": "tachygraphy-microtrext-norm-org/DeBERTa-v3-Base-Cust-LV1-SentimentPolarities-minRegLoss",
        "tokenizer_class": "DebertaV2Tokenizer",
        "model_class": "BERT_architecture",
        "problem_type": "multi_label_classification",
        "base_model": "google/bert-base-uncased",
        "num_labels": 3,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}


class BERT_architecture(nn.Module):

    def __init__(self, bert=AutoModel.from_pretrained("bert-base-uncased",
                                                      device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"))):
        super(BERT_architecture, self).__init__()
        self.bert = bert

        self.dropout = nn.Dropout(0.3)  # Increased dropout for regularization
        self.layer_norm = nn.LayerNorm(768)  # Layer normalization

        self.fc1 = nn.Linear(768, 256)  # Dense layer
        self.fc2 = nn.Linear(256, 3)  # Output layer with 3 classes

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, token_type_ids):
        _, cls_hs = self.bert(sent_id, attention_mask=mask,
                              token_type_ids=token_type_ids, return_dict=False)
        x = self.layer_norm(cls_hs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
