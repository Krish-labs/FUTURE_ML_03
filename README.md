# AI-Powered Customer Support Chatbot: Hybrid Intent-LLM Architecture

## Overview

This project implements a customer support chatbot using a hybrid architecture that combines machine learning-based intent classification with large language model (LLM) response generation.

The system handles routine queries efficiently using a lightweight ML model and delegates complex or ambiguous queries to an LLM.

---

## System Design

### Hybrid Decision Pipeline

The chatbot follows a multi-stage decision flow:

1. Rule-based overrides for specific keywords and edge cases
2. Intent classification using a Scikit-learn pipeline (TF-IDF + Logistic Regression)
3. LLM fallback (Google Gemini / OpenAI API) for complex queries

This approach reduces unnecessary LLM usage while maintaining response quality.

---

## Key Components

### Intent Classifier

* Model: Logistic Regression
* Vectorization: TF-IDF
* Trained on Twitter Customer Support dataset

### Context Handling

* Session-based state tracking using Streamlit
* Enables multi-turn interactions (e.g., handling order ID after status query)

### Response Generation

* Rule-based responses for structured intents
* LLM-based responses for open-ended queries

---

## Tech Stack

* Python
* Scikit-learn
* Streamlit
* Google Gemini API / OpenAI API
* Pandas, NumPy

---

## Project Structure

```text
FUTURE_ML_03/
├── app.py
├── chatbot_logic.py
├── preprocess.py
├── train_classifier.py
├── data/
│   └── sample_twcs.csv   # Sample dataset (full dataset not included)
├── models/
│   └── intent_model.pkl
└── requirements.txt
```

---

## Dataset

This project uses the **Twitter Customer Support dataset (Kaggle)**.

Due to file size limitations, the full dataset is not included in this repository.

Download it from:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

(Optional) A smaller sample dataset can be included for demonstration purposes.

---

## Installation

```bash
git clone https://github.com/your-username/FUTURE_ML_03.git
cd FUTURE_ML_03

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Steps:

1. Enter your API key in the sidebar
2. Start interacting with the chatbot

---

## Limitations

* Intent classification depends on training data and rule-based logic
* The model may misclassify ambiguous queries
* Requires an API key for LLM-based responses
* No integration with real backend systems

---

## Future Improvements

* Replace Logistic Regression with transformer-based models
* Expand dataset with more diverse examples
* Integrate real backend services (order tracking, refunds)

---

## Professional Summary

Developed a customer support chatbot using a hybrid architecture combining machine learning-based intent classification with LLM-driven response generation, along with session-based context handling for multi-turn interactions.
