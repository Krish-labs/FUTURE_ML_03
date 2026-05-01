import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re

# Define intents and associated keywords for heuristic labeling
INTENT_KEYWORDS = {
    'greeting': ['hi', 'hello', 'hey', 'morning', 'afternoon', 'greetings'],
    'refund': ['refund', 'money back', 'return', 'reimbursement', 'cancelled'],
    'status': ['status', 'tracking', 'where is my', 'delivery', 'order', 'package'],
    'complaint': ['broken', 'worst', 'terrible', 'fix', 'awful', 'angry', 'disappointed', 'fail', 'not helpful', 'not satisfied'],
    'technical': ['error', 'crash', 'login', 'password', 'website', 'app', 'not working', 'support', 'contact', 'customer care'],
    'general': ['how to', 'question', 'info', 'information', 'help', 'can you', 'need help']
}

# Manual high-quality training data
MANUAL_DATA = [
    # Greeting
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("hola", "greeting"),
    ("hi assistant", "greeting"),
    ("morning", "greeting"),
    ("hello chatbot", "greeting"),
    ("hey", "greeting"),
    ("hi team", "greeting"),
    ("howdy", "greeting"),
    ("greetings", "greeting"),
    ("hi how are you", "greeting"),
    ("hello can you help me", "greeting"),
    ("hi can you assist", "greeting"),
    ("good evening", "greeting"),
    ("hey assistant", "greeting"),
    ("hi I have a question", "greeting"),
    ("hello there how are you", "greeting"),

    # Order Status
    ("where is my order", "status"),
    ("track my order", "status"),
    ("order status please", "status"),
    ("check my delivery", "status"),
    ("when will my package arrive", "status"),
    ("shipping update", "status"),
    ("my order is delayed", "status"),
    ("tracking number info", "status"),
    ("where is my package", "status"),
    ("delivery status", "status"),
    ("is my order shipped", "status"),
    ("order tracking", "status"),
    ("stuck in transit", "status"),
    ("expected delivery date", "status"),
    ("tracking says delivered but not here", "status"),
    ("has my package left", "status"),
    ("where's my stuff", "status"),
    ("i want to track a package", "status"),
    ("status of my shipment", "status"),
    ("delay in delivery", "status"),

    # Refund
    ("I want a refund", "refund"),
    ("return my money", "refund"),
    ("cancel my order", "refund"),
    ("need a refund", "refund"),
    ("money back guarantee", "refund"),
    ("how to get a refund", "refund"),
    ("i want to return an item", "refund"),
    ("refund policy", "refund"),
    ("cancel this purchase", "refund"),
    ("not what I ordered refund please", "refund"),
    ("reimbursement for my order", "refund"),
    ("give me my money back", "refund"),
    ("order cancellation", "refund"),
    ("i want to cancel", "refund"),
    ("refund for broken item", "refund"),
    ("requesting a refund", "refund"),
    ("processing my refund", "refund"),
    ("i want to return this", "refund"),
    ("getting my money back", "refund"),
    ("can i cancel my subscription", "refund"),

    # Complaint
    ("I want to raise a complaint", "complaint"),
    ("file a complaint", "complaint"),
    ("this is bad service", "complaint"),
    ("I am not satisfied", "complaint"),
    ("worst experience", "complaint"),
    ("i'm very angry", "complaint"),
    ("your app is terrible", "complaint"),
    ("this is unacceptable", "complaint"),
    ("horrible customer support", "complaint"),
    ("i want to report an issue", "complaint"),
    ("totally disappointed", "complaint"),
    ("the quality is poor", "complaint"),
    ("bad experience", "complaint"),
    ("extremely unhappy", "complaint"),
    ("this is a scam", "complaint"),
    ("not helpful at all", "complaint"),
    ("i am complaining about", "complaint"),
    ("awful service", "complaint"),
    ("poor communication", "complaint"),
    ("i don't think you can help", "complaint"),
    ("can't help me", "complaint"),

    # Technical
    ("app is not working", "technical"),
    ("I have a bug", "technical"),
    ("system error", "technical"),
    ("login issues", "technical"),
    ("cannot sign in", "technical"),
    ("website is down", "technical"),
    ("reset my password", "technical"),
    ("error code 404", "technical"),
    ("page not loading", "technical"),
    ("app keeps crashing", "technical"),
    ("technical support needed", "technical"),
    ("button not clicking", "technical"),
    ("slow performance", "technical"),
    ("data sync error", "technical"),
    ("account locked", "technical"),
    ("troubleshooting help", "technical"),
    ("installation failed", "technical"),
    ("software glitch", "technical"),
    ("how to fix app", "technical"),
    ("it's not responding", "technical"),

    # General
    ("I need help", "general"),
    ("can you assist me", "general"),
    ("how does this work", "general"),
    ("tell me more", "general"),
    ("general inquiry", "general"),
    ("i have a question", "general"),
    ("just curious", "general"),
    ("help me please", "general"),
    ("what can you do", "general"),
    ("show me options", "general"),
    ("more information", "general"),
    ("asking for help", "general"),
    ("can you help", "general"),
    ("i am looking for", "general"),
    ("explain this", "general"),
    ("support please", "general"),
    ("need guidance", "general"),
    ("information about", "general"),
    ("how to use", "general"),
    ("basics", "general")
]

def label_intent(text):
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in str(text).lower() for keyword in keywords):
            return intent
    return 'other'

def train_model(input_file, model_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Clean NaN values
    df = df.dropna(subset=['query'])
    
    # Heuristically label a subset
    sample_size = 200000
    df_sample = df.sample(min(len(df), sample_size), random_state=42).copy()
    
    print("Labeling intents heuristically...")
    df_sample['intent'] = df_sample['query'].apply(label_intent)
    
    # Add manual data
    manual_df = pd.DataFrame(MANUAL_DATA, columns=['query', 'intent'])
    df_final = pd.concat([df_sample, manual_df], ignore_index=True)
    
    X = df_final['query'].astype(str)
    y = df_final['intent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training pipeline (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"Model accuracy on test set: {score:.4f}")
    
    print(f"Saving model to {model_file}...")
    joblib.dump(pipeline, model_file)
    print("Model training complete.")

if __name__ == "__main__":
    train_model('processed_qa.csv', 'intent_model.pkl')
