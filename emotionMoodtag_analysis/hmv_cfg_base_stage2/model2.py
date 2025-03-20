from safetensors.torch import save_file, safe_open
from huggingface_hub import hf_hub_download
import json
import safetensors
from transformers import DebertaV2Model, DebertaV2Tokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch
import joblib
import importlib.util
from imports import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))


# from safetensors import load_file, save_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE2 = os.path.join(BASE_DIR, "..", "config", "stage2_models.json")


MODEL_OPTIONS = {
    "2": {
        "name": "DeBERTa v3 Base Custom Model with minimal Regularized Loss",
        "type": "db3_base_custom",
        "module_path": "hmv_cfg_base_stage2.model2",
        "hf_location": "tachygraphy-microtrext-norm-org/DeBERTa-v3-Base-Cust-LV2-EmotionMoodtags-minRegLoss",
        "tokenizer_class": "DebertaV2Tokenizer",
        "model_class": "EmotionModel",
        "problem_type": "regression",
        "base_model": "microsoft/deberta-v3-base",
        "base_model_class": "DebertaV2Model",
        "num_labels": 7,
        "device": "cpu",
        "load_function": "load_model",
        "predict_function": "predict"
    }
}


class EmotionModel(nn.Module):
    def __init__(self, roberta_model, n_classes = 7, dropout_rate = 0.2):
        super(EmotionModel, self).__init__()

        self.roberta = roberta_model
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = output.last_hidden_state
        
        # Extract the [CLS] token representation (first token in the sequence)
        cls_token_state = output.last_hidden_state[:, 0, :]
        output = self.drop(cls_token_state)
        output = self.relu(self.fc1(output))
        output = self.drop(output)
        output = self.relu(self.fc2(output))
#         output = self.drop(output)
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
        # """Loads and caches the model (RoBERTa + EmotionModel) only when called."""
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
    model = EmotionModel.load_pretrained(hf_location)
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
    # probabilities = torch.sigmoid(outputs).cpu().numpy()
    # probabilities = outputs.cpu().numpy()

    relu_logits = F.relu(outputs)
    clipped_logits = torch.clamp(relu_logits, max=1.00000000, min=0.00000000)
    probabilities = clipped_logits.cpu().numpy()

    return probabilities


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully.")
