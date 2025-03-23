from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
from imports import *
import torch.nn as nn
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), )))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_STAGE3 = os.path.join(BASE_DIR, "..", "config", "stage3_models.json")


MODEL_OPTIONS = {
    "3": {
        "name": "Google T5 v1.1 Base for Conditional Text Generation",
        "type": "hf_automodel_finetuned_gt5tctg",
        "module_path": "hmv_cfg_base_stage3.model3",
        "hf_location": "tachygraphy-microtext-normalization-iemk/T5-1.1-HF-seq2seq-Trainer-Batch4",
        "tokenizer_class": "T5Tokenizer",
        "model_class": "T5ForConditionalGeneration",
        "problem_type": "text_transformamtion_and_normalization",
        "base_model": "google/t5-v1_1-base",
        "base_model_class": "T5ForConditionalGeneration",
        "device": "cpu",
        "max_top_k": 32128,
        "load_function": "load_model",
        "predict_function": "predict"
    }
}

model_key = "3"
model_info = MODEL_OPTIONS[model_key]
hf_location = model_info["hf_location"]

tokenizer_class = globals()[model_info["tokenizer_class"]]
model_class = globals()[model_info["model_class"]]


@st.cache_resource
def load_model():
    tokenizer = tokenizer_class.from_pretrained(hf_location)
    print("Loading model 3")
    model = model_class.from_pretrained(hf_location, use_safetensors=True,
                                        # device_map=torch.device(
                                        #     "cuda" if torch.cuda.is_available() else "cpu")
                                        )
    print("Model 3 loaded")

    return model, tokenizer


def predict(
    model, tokenizer, text, device, 
    num_return_sequences=1, 
    beams=None,  # Beam search
    do_sample=False,  # Sampling flag
    temp=None,  # Temperature (only for sampling)
    top_p=None, 
    top_k=None, 
    max_new_tokens=1024, 
    early_stopping=True
):
    # Tokenize input
    padded = tokenizer(text, return_tensors='pt', truncation=False, padding=True).to(device)
    input_ids = padded['input_ids'].to(device)
    attention_mask = padded['attention_mask'].to(device)

    # Validate arguments
    if beams is not None and do_sample:
        raise ValueError("Cannot use `beams` and `do_sample=True` together. Choose either beam search (`beams=5`) or sampling (`do_sample=True, temp=0.7`).")
    
    if temp is not None and not do_sample:
        raise ValueError("`temp` (temperature) can only be used in sampling mode (`do_sample=True`).")

    if (top_p is not None or top_k is not None) and not do_sample:
        raise ValueError("`top_p` and `top_k` can only be used in sampling mode (`do_sample=True`).")

    # Beam search (Deterministic)
    if beams is not None:
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences, 
            num_beams=beams, 
            early_stopping=early_stopping, 
            do_sample=False  # No randomness
        )

    # Sampling Cases
    else:
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": True,  # Enable stochastic sampling
            "temperature": temp if temp is not None else 0.7,  # Default temp if not passed
        }

        # Add `top_p` if set
        if top_p is not None:
            generate_args["top_p"] = top_p

        # Add `top_k` if set
        if top_k is not None:
            generate_args["top_k"] = top_k

        # Generate
        outputs = model.generate(**generate_args)

    # Decode predictions into human-readable text
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return predictions
