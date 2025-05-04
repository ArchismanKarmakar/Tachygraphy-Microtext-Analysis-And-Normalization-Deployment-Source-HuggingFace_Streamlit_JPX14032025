---
title: Tachygraphy based Microtext Analysis And Normalization
emoji: 💻
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.44.0
python_version: "3.12"
app_file: app_main_hf.py
pinned: true
short_description: This project focuses on the Tachygraphy Microtext Analysis
tags:
  - machine-learning
  - deep-learning
  - tachygraphy
  - text2text
  - microtext
  - nlp
  - research
  - colorful
  - demo
---


<!-- ---
title: Tachygraphy Microtext Analysis And Normalization
emoji: 💻
colorFrom: purple
colorTo: gray
sdk: docker
app_file: app_main_hf.py
pinned: false
short_description: This project focuses on the Tachygraphy Microtext Analysis
--- -->

# Tachygraphy-Microtext-Analysis-And-Normalization-Deployment-Source-HuggingFace_Streamlit_JPX14032025


# Tachygraphy based Micro-text Analysis And Normalization
<!---
---
title: "Tachygraphy based Micro-text Analysis & Normalization"
emoji: "⚡"
colorFrom: "pink"
colorTo: "blue"
sdk: "static"
pinned: false
---
--->

<!-- ---
title: README
emoji: 😻
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
---
 -->
 
<div align="center">
  
<!-- ![Project Logo](https://via.placeholder.com/150) -->

# Tachygraphy based Micro-text Analysis & Normalization

*Welcome to the Tachygraphy based Micro-text Analysis & Normalization Project. This page outlines our project’s key stages, sources, sample analysis examples, and team information.*

</div>

---

## Dashboard

### Project Stages

1. **Sentiment Polarity Analysis**
2. **Emotion Mood-tag Analysis**
3. **Text Transformation & Normalization**
4. **Stacked all 3 stages with their best models**
5. **Data Correction & Collection**

### Sources & Deployment Links

| **Training Source** | **Kaggle Collections** | **Hugging Face Org** |
| ------------------- | ---------------------- | -------------------- |
| [GitHub @ Tachygraphy Micro-text Analysis & Normalization](https://github.com/ArchismanKarmakar/Tachygraphy-Microtext-Analysis-And-Normalization) | [Kaggle Dataset](https://www.kaggle.com/datasets/archismancoder/dataset-tachygraphy/data?select=Tachygraphy_MicroText-AIO-V3.xlsx) | [Hugging Face @ Tachygraphy Micro-text Normalization](https://huggingface.co/Tachygraphy-Microtext-Normalization-IEMK25) |

| **Deployment Source** | **Streamlit Deployment** | **Hugging Face Space Deployment** |
| --------------------- | ------------------------ | --------------------------------- |
| [GitHub Deployment Repo](https://github.com/ArchismanKarmakar/Tachygraphy-Microtext-Analysis-And-Normalization-Deployment-Source-HuggingFace_Streamlit_JPX14032025) | [Streamlit App](https://tachygraphy-microtext.streamlit.app/) | [Hugging Face Space](https://huggingface.co/spaces/Tachygraphy-Microtext-Normalization-IEMK25/Tachygraphy-Microtext-Analysis-and-Normalization-ArchismanCoder) |

---

## Project Overview

Tachygraphy—originally developed to expedite writing—has evolved over centuries. In the 1990s, it reappeared as micro‑text, driving faster communication on social media with its “Anytime, Anyplace, Anybody, and Anything (4A)” characteristic. This project focuses on the analysis and normalization of micro‑text (the prevalent informal communication today) to improve NLP tasks such as sentiment analysis, emotion detection, and overall text transformation for clear 4A message decoding.

---


### Sample Example 1
```mermaid
graph TD;
    %% Input and normalized text nodes
    A["Input Text: i don't know fr y he's sooo sad"]
    B["Normalized Text: i do not know for real why he's so sad"]
    C["Sentiment"]

    A --> B
    A -->|Sentiment| C

    %% Sentiment value nodes (values inside the boxes)
    C -->|Negative| D["0.99587"]
    C -->|Neutral| E["6.23e-05"]
    C -->|Positive| F["2.10e-05"]

    %% Converge sentiment nodes to Emotion stage
    D -->|Emotion| G
    E -->|Emotion| G
    F -->|Emotion| G

    G["Emotion"]

    %% Emotion nodes: arrow labels show emotion category; node boxes show numeric values.
    G -->|Anger| H["0.0"]
    G -->|Disgust| I["0.0"]
    G -->|Fear| J["0.01028"]
    G -->|Joy| K["0.0"]
    G -->|Neutral| L["0.02194"]
    G -->|Sadness| M["1.0"]
    G -->|Surprise| N["0.02158"]
    A -->|Emotion| G

%% Style the Neutral and Positive sentiment arrows with a lighter stroke.
linkStyle 6 stroke:#cccccc, stroke-width:1px;
linkStyle 7 stroke:#cccccc, stroke-width:1px;

```

### Sample Example 2
```mermaid
graph LR;
    %% Input and normalized text nodes
    A["Input Text: you rlly think all that talk means u tough? lol, when I step up, u ain't gon say sh*t"]
    B["Normalized Text: you really think all that talk makes you tough [lol](laughed out loud) when i step up you are not going to say anything"]
    C["Sentiment"]

    A --> B
    A -->|Sentiment| C

    %% Sentiment value nodes
    C -->|Negative| D["0.99999"]
    C -->|Neutral| E["6.89e-06"]
    C -->|Positive| F["1.11e-05"]

    %% Converge sentiment nodes to Emotion stage
    D -->|Emotion| G
    E -->|Emotion| G
    F -->|Emotion| G

    G["Emotion"]

    %% Emotion nodes: arrow labels show emotion category; nodes show numeric values.
    G -->|Anger| H["0.14403"]
    G -->|Disgust| I["0.03928"]
    G -->|Fear| J["0.01435"]
    G -->|Joy| K["0.04897"]
    G -->|Neutral| L["0.49485"]
    G -->|Sadness| M["0.02111"]
    G -->|Surprise| N["0.23741"]
    A -->|Emotion| G

%% Style the Neutral and Positive sentiment arrows with a lighter stroke.
linkStyle 6 stroke:#cccccc, stroke-width:1px;
linkStyle 7 stroke:#cccccc, stroke-width:1px;
```

### Sample Example 3
```mermaid
graph TD;
    %% Input and normalized text nodes
    A["Input Text: bruh, floods in Kerala, rescue ops non‑stop 🚁"]
    B["Normalized Text: Brother, the floods in Kerala are severe, and rescue operations are ongoing continuously."]
    C["Sentiment"]

    A --> B
    A -->|Sentiment| C

    %% Sentiment value nodes
    C -->|Negative| D["4.44e-05"]
    C -->|Neutral| E["0.99989"]
    C -->|Positive| F["7.10e-05"]

    %% Converge sentiment nodes to Emotion stage
    D -->|Emotion| G
    E -->|Emotion| G
    F -->|Emotion| G

    G["Emotion"]

    %% Emotion nodes: arrow labels show emotion category; node boxes show numeric values.
    G -->|Anger| H["0.08018"]
    G -->|Disgust| I["0.01526"]
    G -->|Fear| J["0.60187"]
    G -->|Joy| K["0.00411"]
    G -->|Neutral| L["0.02194"]
    G -->|Sadness| M["1.0"]
    G -->|Surprise| N["0.02158"]
    A -->|Emotion| G

%% Style the Neutral and Positive sentiment arrows with a lighter stroke.
linkStyle 5 stroke:#cccccc, stroke-width:1px;
linkStyle 7 stroke:#cccccc, stroke-width:1px;

```

