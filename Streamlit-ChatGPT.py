from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

st.set_page_config(page_title="Helix Shop Assistant", page_icon="🛍️")

st.title("🛍️ Helix Shop Assistant")
st.write("Your AI shopping assistant — product help, order questions, and recommendations.")

DEFAULT_SYSTEM_PROMPT = """You are Helix, a friendly and knowledgeable e-commerce shopping assistant for an online store.

Your job is to:
- Help customers find products that match what they're looking for
- Answer questions about pricing, sizing, materials, and availability
- Assist with order status, shipping, returns, and exchanges
- Recommend complementary or alternative products
- Keep a warm, helpful, sales-friendly tone without being pushy

If you don't have specific info about a product or order (e.g. real-time stock or tracking numbers), say so honestly and suggest the customer check their account or contact support instead of guessing.
"""

with st.sidebar:
    st.header("Store Settings")
    store_name = st.text_input("Store name", value="Helix Store")
    system_message = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=220)
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-5.2"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    if st.button("Clear chat"):
        st.session_state.messages = []

chat = ChatOpenAI(model_name=model_name, temperature=temperature)

# Initialize / keep system prompt in sync with sidebar edits
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_message)]
else:
    st.session_state.messages[0] = SystemMessage(content=system_message)

# Render prior turns
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input — only fires once per submission, unlike text_input
user_input = st.chat_input(f"Ask {store_name} anything...")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Helix is thinking..."):
            response = chat.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))