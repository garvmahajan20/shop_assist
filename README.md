# ShopAssist: NLP-Based E-commerce Intent Classification

ShopAssist is an NLP-based e-commerce customer support chatbot prototype that classifies customer inquiries into 19 predefined intents using TF-IDF feature extraction and a Feedforward Neural Network (FNN). It incorporates softmax probability scoring, confidence-threshold filtering, and an intent-based response routing mechanism backed by mock order data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture & Workflow](#architecture--workflow)
- [Model Architecture](#model-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training & Evaluation](#training--evaluation)
- [Repository Files & Saved Artifacts](#repository-files--saved-artifacts)
- [Installation & Usage](#installation--usage)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [License & Attribution](#license--attribution)

---

## Project Overview

Customer inquiries in e-commerce platforms often follow recurring patterns—such as requesting delivery updates, asking about return policies, or reporting missing items. ShopAssist explores a lightweight, efficient machine learning pipeline for automated intent recognition. 

Rather than relying on heavy transformer architectures or external API calls for basic categorization, ShopAssist demonstrates that a well-tuned TF-IDF representation combined with a multi-layer Feedforward Neural Network (FNN) can achieve high classification accuracy on standardized e-commerce queries while maintaining near-instant inference speed and minimal compute requirements.

---

## Key Features

- **TF-IDF Text Feature Extraction:** Converts customer natural language queries into a high-dimensional numerical representation (2,597 features).
- **FNN-Based Multiclass Intent Classification:** Deep Feedforward Neural Network trained to distinguish between 19 domain-specific e-commerce intents.
- **Softmax Probability Calculation:** Converts raw network logits into calibrated probability distributions across all candidate intents.
- **Confidence-Threshold Handling:** Applies an 80% (`>= 0.80`) confidence threshold to safeguard against uncertain predictions, prompting the user for clarification when confidence is low.
- **Saved Model & Preprocessing Artifacts:** Persists the trained model weights, fitted TF-IDF vocabulary, and intent labels for immediate out-of-the-box inference without retraining.
- **Intent-Based Response Routing:** Routes recognized intents to specialized handler functions.
- **Mock Order Lookup & Order Operations:** Demonstrates operational execution (e.g., order tracking, cancellation, return initiation) using simulated order records.

---

## Architecture & Workflow

The end-to-end inference workflow operates as follows:

```
Customer Message
       │
       ▼
[Input Validation] ── (Check for empty text / whitespace)
       │
       ▼
[TF-IDF Transformation] ── (Transform input using fitted vocabulary of 2,597 terms)
       │
       ▼
[FNN Model Prediction] ── (Forward pass producing 19 raw linear logits)
       │
       ▼
[Softmax Probability Layer] ── (Compute class probabilities)
       │
       ▼
[Confidence Threshold] ── (max(P) >= 0.80?)
      ├── No  ──► Prompt user to clarify or rephrase message
      └── Yes ──► Extract predicted class index
                    │
                    ▼
          [Intent Mapping] ── (Map index to intent via selected_intent.json)
                    │
                    ▼
          [Response Router] ──► Fetch mock order details / return policy response
```

---

## Model Architecture

The classifier is implemented as a sequential Feedforward Neural Network (FNN) in TensorFlow/Keras:

- **Input Layer:** Accepts a 2,597-dimensional sparse TF-IDF feature vector (`shape=(2597,)`).
- **Dense Hidden Layer 1:** 128 units, ReLU activation function (`332,544` parameters).
- **Dense Hidden Layer 2:** 128 units, ReLU activation function (`16,512` parameters).
- **Dense Hidden Layer 3:** 64 units, ReLU activation function (`8,256` parameters).
- **Output Layer:** 19 units, linear activation generating unnormalized logits (`1,235` parameters).
- **Inference Activation:** `tf.nn.softmax` applied over the 19 output logits to generate prediction probabilities.

### Parameter Summary

| Layer (type) | Output Shape | Param # |
|---|---|---|
| `Input` | `(None, 2597)` | 0 |
| `Dense` (ReLU) | `(None, 128)` | 332,544 |
| `Dense` (ReLU) | `(None, 128)` | 16,512 |
| `Dense` (ReLU) | `(None, 64)` | 8,256 |
| `Dense` (Linear logits) | `(None, 19)` | 1,235 |

- **Total Trainable Parameters:** 358,547 (~1.37 MB)
- **Total Model Parameters (including optimizer states):** 1,075,643 (~4.10 MB)

---

## Dataset & Preprocessing

The model is trained on utterances from the [Bitext Retail / E-commerce LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset), an industry-standard benchmark dataset designed for customer support chatbot development.

### Intent Selection & Filtering

The original dataset covers a broad variety of conversational scenarios. ShopAssist isolates 19 core intents directly applicable to retail and order management:

1. `track_order`
2. `cancel_order`
3. `order_history`
4. `delivery_issue`
5. `delivery_time`
6. `missing_item`
7. `damaged_delivery`
8. `refund_status`
9. `refund_policy`
10. `return_policy`
11. `return_product`
12. `request_refund`
13. `add_product`
14. `remove_product`
15. `availability`
16. `product_information`
17. `payment_issue`
18. `payment_methods`
19. `human_agent`

Filtering for these 19 intents yields **18,433 labeled samples**.

### Train / Validation / Test Split

A stratified split was performed to preserve class distributions across all subsets using `random_state=1`:

- **Training set (80%):** 14,746 samples
- **Validation set (10%):** 1,843 samples
- **Test set (10%):** 1,844 samples

### Feature Extraction

A `TfidfVectorizer` from `scikit-learn` was fit exclusively on the training corpus (`X_train`) to prevent data leakage, yielding a vocabulary of **2,597 unique n-gram features**. The validation and test sets were transformed using this fitted vectorizer.

---

## Training & Evaluation

### Training Setup

- **Loss Function:** `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer:** `Adam(learning_rate=0.01)`
- **Epochs:** 25
- **Batch Size:** Default Keras batch size (32)

### Verification Metrics

The following metrics are recorded from the completed training run in [model.ipynb](model.ipynb):

- **Final Training Accuracy:** 100.00% (Loss: `6.57e-08`)
- **Validation Evaluation (`model.evaluate`):**
  - **Loss:** `0.0393`
  - **Accuracy:** `99.57%`
- **Test Evaluation (`model.evaluate`):**
  - **Loss:** `0.0385`
  - **Accuracy:** `99.30%`

> **Note on Generalization:** While the model achieves >99% test accuracy on this dataset, these metrics reflect evaluation on synthetic and semi-synthetic benchmark utterances. Strong benchmark performance does not automatically guarantee equivalent generalization to noisy, unstructured real-world customer text containing typos, complex grammatical shifts, or out-of-scope requests.

---

## Repository Files & Saved Artifacts

```text
shop_assist/
├── README.md               # Project documentation and reproduction guide
├── requirements.txt        # Python dependency specifications
├── .gitignore              # Ignored patterns for clean version control
├── model.ipynb             # Complete ML workflow notebook (training & evaluation)
├── shopassist_model.keras  # Saved Keras model architecture and trained weights
├── tfidf_vectorizer.pkl    # Fitted scikit-learn TF-IDF vectorizer
└── selected_intent.json    # Ordered list of 19 intent labels mapping indices to classes
```

### Purpose of Model Artifacts

To enable standalone inference without rerunning the training notebook, three persistent artifacts are provided:

1. **`shopassist_model.keras` (~4.1 MB):** Contains the complete trained Keras model, including layer architecture, activation configurations, and learned weights.
2. **`tfidf_vectorizer.pkl` (~54 KB):** Contains the serialized `TfidfVectorizer` instance with the exact 2,597-term vocabulary and IDF weights fitted on the training split. New text inputs must be transformed using this vectorizer to match the input space expected by the neural network.
3. **`selected_intent.json` (~330 B):** An ordered JSON array defining the index-to-intent mapping (0 to 18) used to translate the model's argmax class index into a human-readable intent label.

All three artifacts must be loaded together to execute the inference pipeline.

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/garvmahajan20/shop_assist.git
cd shop_assist
```

### 2. Set Up a Virtual Environment & Install Dependencies

It is recommended to use Python 3.10 to 3.13:

```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Open the Notebook

```bash
jupyter notebook model.ipynb
```

You can view the full data exploration, feature extraction, training progression, and evaluation outputs directly in the notebook without needing to rerun training.

### 4. Running Standalone Inference (Python Example)

You can load the saved artifacts and test predictions directly in Python:

```python
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. Load artifacts
model = keras.models.load_model("shopassist_model.keras")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
with open("selected_intent.json", "r") as f:
    selected_intents = json.load(f)

# 2. Sample customer query
query = "Can you tell me where my order is right now?"

# 3. Preprocess and predict
query_vec = vectorizer.transform([query.strip()])
output_logits = model.predict(query_vec)
output_prob = tf.nn.softmax(output_logits).numpy()

confidence = float(np.max(output_prob))
predicted_idx = int(np.argmax(output_prob))
predicted_intent = selected_intents[predicted_idx]

# 4. Confidence check (threshold: 0.80)
if confidence >= 0.80:
    print(f"Predicted Intent: {predicted_intent} (Confidence: {confidence:.2%})")
else:
    print(f"Low confidence ({confidence:.2%}). Please clarify your message.")
```

---

## Limitations & Future Improvements

- **Mock Data Dependency:** The response router operates on simulated in-memory order records (`mock_orders`). It is not integrated with a real database or live e-commerce API.
- **Fixed Intent Scope:** The model is bounded to the 19 predefined intents. Out-of-vocabulary or out-of-scope customer queries that do not match these categories must be handled by confidence rejection or fallback routing.
- **Confidence Threshold Limitations:** An 80% confidence cutoff reduces false positives on ambiguous queries, but softmax probabilities on high-dimensional linear boundaries can still occasionally exhibit overconfidence on out-of-distribution inputs.
- **Generalization Assessment:** Further evaluation against uncurated, human-written conversational transcripts is recommended before deploying in production environments.
- **Future Enhancements:**
  - Transitioning to subword embeddings or fine-tuned compact transformer models (e.g., DistilBERT) for improved handling of typos and complex phrasing.
  - Integration with an actual REST/GraphQL order management API.
  - Incorporating multi-turn conversational dialogue management.

---

## License & Attribution

- **Dataset:** The [Bitext Retail / E-commerce LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset) is identified on its official dataset card as licensed under the **Community Data License Agreement (CDLA) Sharing 1.0**. See the [official dataset card](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset/blob/main/README.md).
- **Attribution and sharing:** The dataset card describes attribution and share-alike requirements for the dataset and its derivatives. Review and comply with the [CDLA-Sharing 1.0 terms](https://cdla.dev/sharing-1-0/) when using or redistributing dataset-derived materials.
- **Code and model artifacts:** The dataset license should not be presented as automatically licensing every part of this repository. This README does not declare a separate project-wide license for the code and model artifacts.
