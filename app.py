import streamlit as st
import os
from chatbot_logic import ChatbotLogic

st.set_page_config(page_title="AI Customer Support", layout="wide")

st.title("🤖 AI-Powered Customer Support")

# Sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

st.sidebar.markdown("### Supported Intents:")
st.sidebar.write("- Greeting")
st.sidebar.write("- Refund Request")
st.sidebar.write("- Order Status")
st.sidebar.write("- Complaint")
st.sidebar.write("- Technical Support")
st.sidebar.write("- General Inquiry")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_intent = None
    st.session_state.order_id = None

# Initialize chatbot
if "bot" not in st.session_state:
    st.session_state.bot = ChatbotLogic()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "order_id" not in st.session_state:
    st.session_state.order_id = None

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("How can I help you today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Check for context-aware follow-up
    if user_input.isdigit() and st.session_state.last_intent in ["status", "refund"]:
        intent = st.session_state.last_intent
        st.session_state.order_id = user_input
        if intent == "status":
            response = f"Thanks! I'm checking the status for Order ID {user_input}. Your order is currently in transit 🚚."
        else:
            response = f"Thanks! I've received your Order ID {user_input}. I'm processing your refund request now. You will receive an email confirmation shortly."
    else:
        response, intent = st.session_state.bot.get_response(
            user_input, st.session_state.messages
        )

    # Store last intent
    st.session_state.last_intent = intent

    bot_reply = f"**[Intent: {intent}]**\n\n{response}"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)