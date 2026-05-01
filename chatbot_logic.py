import joblib
import os
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ChatbotLogic:
    def __init__(self, model_path='intent_model.pkl'):
        self.model = joblib.load(model_path)
        self.api_key = None
        self.analyzer = SentimentIntensityAnalyzer()

    def predict_intent(self, text):
        text_lower = text.lower()
        words = text_lower.split()

        # 🔥 Rule-based overrides (priority)

        # Greeting (use split words for short greeting tokens to avoid substring matches)
        if any(word in words for word in ["hi", "hello", "hey", "morning", "afternoon", "greetings"]):
            return "greeting"

        # Complaint
        if any(phrase in text_lower for phrase in [
            "complaint", "raise a complaint", "not satisfied", "bad service", 
            "worst", "unacceptable", "disappointed", "poor quality", "scam"
        ]):
            return "complaint"

        # Technical Support
        if any(word in text_lower for word in [
            "support", "technical", "error", "bug", "crash", "not working", "login issue"
        ]):
            return "technical"

        # Refund
        if any(word in text_lower for word in [
            "refund", "return money", "cancel order", "get my money back", "cancellation"
        ]):
            return "refund"

        # Order Status
        if any(word in text_lower for word in [
            "order status", "track", "delivery", "where is my", "package status"
        ]):
            return "status"

        # Handle negation / Sentiment cases
        if any(phrase in text_lower for phrase in ["don't think", "can't help", "cannot help", "useless"]):
            return "complaint"

        # Sentiment awareness using VADER (secondary check for complaints)
        score = self.analyzer.polarity_scores(text)["compound"]
        if score < -0.5:
            return "complaint"

        # Fallback to ML model
        intent = self.model.predict([text])[0]
        
        # Handle unknown/other intents
        if intent == "other":
            intent = "general"
            
        return intent

    def configure_gemini(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return False
        
        if api_key != self.api_key:
            genai.configure(api_key=api_key)
            self.api_key = api_key
        
        return True
    

    def get_response(self, user_input, history):
        intent = self.predict_intent(user_input)

        # Check API key
        if not self.configure_gemini():
            return self.rule_based_fallback(intent), intent

        system_instruction = (
            "You are a professional customer support assistant. "
            f"User intent: {intent}. "
            "Be helpful, short, and clear. Ask for order ID if needed."
        )

        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )

            gemini_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_input)

            return response.text, intent

        except Exception:
            return self.rule_based_fallback(intent), intent

    def rule_based_fallback(self, intent):
        responses = {
            "greeting": "Hello! How can I assist you today?",
            "refund": "Please provide your order ID to process your refund.",
            "status": "Please share your order ID so I can check the status.",
            "complaint": "I'm sorry for the inconvenience. Please explain your issue.",
            "technical": "Please describe the technical issue you're facing.",
            "general": "How can I help you today?"
        }
        return responses.get(intent, "I'm here to help you with your queries.")