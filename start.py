import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import pytz
import time
import re

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Local RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.chat-container {max-height: 500px; overflow-y: auto; padding: 20px; border-radius: 15px;
background: linear-gradient(180deg, #1e1e1e 0%, #252525 100%); border: 1px solid #444;
box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); margin-bottom: 20px;}
.chat-row {display: flex; margin-bottom: 15px; animation: fadeIn 0.3s ease-in;}
.chat-bubble {padding: 12px 18px; border-radius: 20px; max-width: 70%; word-wrap: break-word;
font-size: 15px; line-height: 1.5;}
.user-bubble {background: linear-gradient(135deg, #005f99 0%, #007acc 100%); color: white;
margin-left: auto; margin-right: 10px;}
.bot-bubble {background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%); color: #e0e0e0;
margin-right: auto; margin-left: 10px;}
.timestamp {font-size: 12px; color: #888; margin: 5px 10px; text-align: right;}
.bot-timestamp {text-align: left;}
.header {display: flex; align-items: center; gap: 10px; margin-bottom: 10px; background: transparent; padding: 10px;}
.header img {width: 40px; height: 40px;}
.header h1 {color: #ffffff; margin: 0;}
.query-section {background: #1e1e1e; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 20px;}
.query-section p {color: #ffffff; font-size: 16px; margin: 0;}
.status-box {padding: 10px; border-radius: 8px; background-color: #2a2a2a; color: white; font-size: 14px;}
@keyframes fadeIn {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);}}
.chat-container::-webkit-scrollbar {width: 8px;}
.chat-container::-webkit-scrollbar-thumb {background-color: #444; border-radius: 8px;}
.chat-container::-webkit-scrollbar-track {background: #1e1e1e;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
    <img src="https://img.icons8.com/ios-filled/50/ffffff/chat.png" alt="Chat Icon"/>
    <h1>Local RAG Chatbot</h1>
</div>
<div class="query-section">
    <p>Ask your queries</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# -----------------------------
# Cleaning function
# -----------------------------
def clean_lab_text(text: str) -> str:
    pattern = re.compile(r"(ng/mL|mg/dL)\s+(\d+\.\d+)\s*-\s*(\d+\.\d+)(\d+\.\d+)")
    def replacer(m):
        unit = m.group(1)
        ref_low = m.group(2)
        ref_high = m.group(3)
        result = m.group(4)
        return f"**Result**: {result} {unit}, **Reference Range**: {ref_low} – {ref_high} {unit}"
    return pattern.sub(replacer, text)

# -----------------------------
# Load PDF or fallback
# -----------------------------
text_data = """
OpenAI develops powerful AI models. Hugging Face hosts open-source models. 
Retrieval-Augmented Generation (RAG) allows chatbots to fetch knowledge from documents. 
This demo runs completely locally using Hugging Face models instead of OpenAI.
Lab values: ng/mL 0.81 - 3.85 0.68
"""
documents = []
if uploaded_file:
    with st.spinner("Processing PDF(s)..."):
        for file in uploaded_file:
            with open(f"temp_{file.name}", "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(f"temp_{file.name}")
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_lab_text(doc.page_content)
            documents.extend(docs)
else:
    documents = [text_data]

# -----------------------------
# Init chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Split into chunks
# -----------------------------
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
if isinstance(documents, list) and hasattr(documents[0], "page_content"):
    chunks = text_splitter.split_documents(documents)
else:
    chunks = text_splitter.split_text(str(documents))

# -----------------------------
# Embeddings + Vector Store
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(
    [c.page_content if hasattr(c, "page_content") else c for c in chunks],
    embeddings
)

# -----------------------------
# Local Hugging Face Model
# -----------------------------
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, temperature=0.1)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -----------------------------
# Conversational Retrieval Chain
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)

# -----------------------------
# Simple responses
# -----------------------------
simple_responses = {
    "thank you": "You're welcome! 😊",
    "thanks": "My pleasure! 😄",
    "hello": "Hi there! How can I help you today? 👋",
    "hi": "Hello! What’s on your mind? 😄",
    "bye": "See you later! Take care! 👋",
    "goodbye": "Goodbye! Feel free to return anytime! 😊"
}

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("💬 Type your message...")
if user_input:
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.now(ist).strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input, current_time))

    with st.spinner("Typing..."):
        # Simple replies first
        user_input_lower = user_input.lower().strip()
        bot_message = simple_responses.get(user_input_lower, None)
        if not bot_message:
            response = qa_chain.invoke({"question": user_input})
            bot_message = response["answer"]

        # (No immediate render here — only save to history)
        st.session_state.chat_history.append(("bot", bot_message, current_time))

# -----------------------------
# Render full chat history (bottom only)
# -----------------------------
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg, timestamp in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div class='chat-row' style='justify-content: flex-end;'>
                    <div class='chat-bubble user-bubble'>{msg}</div>
                </div>
                <div class='timestamp'>{timestamp}</div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='chat-row' style='justify-content: flex-start;'>
                    <div class='chat-bubble bot-bubble'>{msg}</div>
                </div>
                <div class='timestamp bot-timestamp'>{timestamp}</div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)
