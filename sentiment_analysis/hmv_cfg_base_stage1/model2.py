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
from transformers import DebertaV2Model, DebertaV2Tokenizer
import safetensors
# from safetensors import load_file, save_file
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, safe_open

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE1 = os.path.join(BASE_DIR, "..", "config", "stage1_models.json")

MODEL_OPTIONS = {
"2": {
        "name": "DeBERTa v3 Base Custom Model with minimal Regularized Loss",
        "type": "db3_base_custom",
        "module_path": "hmv_cfg_base_stage1.model2",
        "hf_location": "tachygraphy-microtrext-norm-org/DeBERTa-v3-Base-Cust-LV1-SentimentPolarities-minRegLoss",
        "tokenizer_class": "DebertaV2Tokenizer",
        "model_class": "SentimentModel",
        "problem_type": "multi_label_classification",
        "base_model": "microsoft/deberta-v3-base",
        "base_model_class": "DebertaV2Model",
        "num_labels": 3,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}

class SentimentModel(nn.Module):
    def __init__(self, roberta_model, n_classes=3, dropout_rate=0.2):
        super(SentimentModel, self).__init__()

        self.roberta = roberta_model
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_state = output.last_hidden_state[:, 0, :]
        output = self.drop(cls_token_state)
        output = self.relu(self.fc1(output))
        return self.out(output)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        model_weights = self.state_dict()
        save_file(model_weights, os.path.join(save_directory, "model.safetensors"))

        config = {
            "hidden_size": self.roberta.config.hidden_size,
            "num_labels": self.out.out_features,
            "dropout_rate": self.drop.p,
            "roberta_model": self.roberta.name_or_path,  # ✅ Save model name
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        print(f"Model saved in {save_directory}")


    @classmethod
    @st.cache_resource
    def load_pretrained(cls, model_path_or_repo):
        """Loads and caches the model (RoBERTa + SentimentModel) only when called."""
        print(f"Loading model from {model_path_or_repo}...")

        model_config_path = hf_hub_download(model_path_or_repo, "config.json")
        model_weights_path = hf_hub_download(model_path_or_repo, "model.safetensors")

        with open(model_config_path, "r") as f:
            config = json.load(f)

        print(f"Loading RoBERTa model: {config['roberta_model']}...")
        roberta_model = DebertaV2Model.from_pretrained(
            config["roberta_model"],
        )

        model = cls(
            roberta_model, n_classes=config["num_labels"], dropout_rate=config["dropout_rate"]
        )

        with safe_open(model_weights_path, framework="pt", device="cpu") as f:
            model_weights = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(model_weights)

        print(f"Model loaded from {model_path_or_repo}")
        return model
    

model_key = "2"
model_info = MODEL_OPTIONS[model_key]
hf_location = model_info["hf_location"]
base_model = model_info["base_model"]

tokenizer_class = globals()[model_info["tokenizer_class"]]
model_class = globals()[model_info["model_class"]]

    
@st.cache_resource
def load_model():
    tokenizer = tokenizer_class.from_pretrained(hf_location)
    print("Loading model 2")
    model = SentimentModel.load_pretrained(hf_location)
    print("Model 2 loaded")
    # model.eval()

    return model, tokenizer


def predict(text, model, tokenizer, device, max_len=128):
    # model.eval()  # Set model to evaluation mode

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
    probabilities = torch.sigmoid(outputs).cpu().numpy()
    # probabilities = outputs.cpu().numpy()

    return probabilities


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully.")


### COMMENTED CODE ###


# @st.cache_resource

# def load_pretrained(model_path_or_repo):

#     model_config_path = hf_hub_download(model_path_or_repo, "config.json")
#     model_weights_path = hf_hub_download(model_path_or_repo, "model.safetensors")

#     with open(model_config_path, "r") as f:
#         config = json.load(f)

#     roberta_model = DebertaV2Model.from_pretrained(
#         config["roberta_model"],
#     )

#     model = SentimentModel(
#         roberta_model, n_classes=config["num_labels"], dropout_rate=config["dropout_rate"]
#     )

#     with safe_open(model_weights_path, framework="pt", device="cpu") as f:
#         model_weights = {key: f.get_tensor(key) for key in f.keys()}
#     model.load_state_dict(model_weights)

#     print(f"Model loaded from {model_path_or_repo}")
#     return model




# class SentimentModel(nn.Module):
#     def __init__(self, roberta_model=DebertaV2Model.from_pretrained(
#             'microsoft/deberta-v3-base',
#             device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         ), n_classes=3, dropout_rate=0.2):
#         super(SentimentModel, self).__init__()

#         self.roberta = roberta_model
#         self.drop = nn.Dropout(p=dropout_rate)
#         self.fc1 = nn.Linear(self.roberta.config.hidden_size, 256)  # Reduced neurons
#         self.relu = nn.ReLU()
#         self.out = nn.Linear(256, n_classes)

#     def forward(self, input_ids, attention_mask):
#         output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         cls_token_state = output.last_hidden_state[:, 0, :]
#         output = self.drop(cls_token_state)
#         output = self.relu(self.fc1(output))
#         return self.out(output)

#     def save_pretrained(self, save_directory):
#         os.makedirs(save_directory, exist_ok=True)

#         # Save model weights using safetensors
#         model_weights = self.state_dict()
#         save_file(model_weights, os.path.join(save_directory, "model.safetensors"))

#         # Save model config
#         config = {
#             "hidden_size": self.roberta.config.hidden_size,
#             "num_labels": self.out.out_features,
#             "dropout_rate": self.drop.p,
#             "roberta_model": self.roberta.name_or_path
#         }
#         with open(os.path.join(save_directory, "config.json"), "w") as f:
#             json.dump(config, f)

#         print(f"Model saved in {save_directory}")

#     @classmethod
#     def load_pretrained(cls, model_path_or_repo, roberta_model):
#         # if model_path_or_repo.startswith("http") or "/" not in model_path_or_repo:
#         #     # Load from Hugging Face Hub
#         #     model_config_path = hf_hub_download(model_path_or_repo, "config.json")
#         #     model_weights_path = hf_hub_download(model_path_or_repo, "model.safetensors")
#         # else:
#         #     # Load from local directory
#         #     model_config_path = os.path.join(model_path_or_repo, "config.json")
#         #     model_weights_path = os.path.join(model_path_or_repo, "model.safetensors")

#         model_config_path = hf_hub_download(model_path_or_repo, "config.json")
#         model_weights_path = hf_hub_download(model_path_or_repo, "model.safetensors")

#         # Load model config
#         with open(model_config_path, "r") as f:
#             config = json.load(f)

#         # Load RoBERTa model
#         roberta_model = DebertaV2Model.from_pretrained(config["roberta_model"])

#         # Initialize SentimentModel
#         model = cls(
#             roberta_model,
#             n_classes=config["num_labels"],
#             dropout_rate=config["dropout_rate"]
#         )

#         # Load safetensors weights
#         with safe_open(model_weights_path, framework="pt", device="cpu") as f:
#             model_weights = {key: f.get_tensor(key) for key in f.keys()}
#         model.load_state_dict(model_weights)

#         print(f"Model loaded from {model_path_or_repo}")
#         return model
