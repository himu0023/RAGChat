# Local RAG Chatbot

A Streamlit-based app using Retrieval-Augmented Generation (RAG) to create a locally hosted conversational AI. Powered by Hugging Face's open-source models, it processes queries from uploaded PDFs or a default text dataset. With a sleek, dark-themed UI, it shows user messages in deep blue on the right and bot replies in black on the left, featuring a typing effect and IST timestamps (e.g., 15:45:00, Aug 19, 2025). It supports multi-PDF uploads, limits chat history to 50 messages, and allows clearing or downloading conversations. Simple inputs like "thank you" trigger friendly responses, while RAG handles document-based queries effectively. This offline, privacy-focused demo highlights practical AI.

## Features
- Locally hosted AI with privacy focus
- RAG pipeline for document-based queries
- Multi-PDF upload support
- Typing effect with IST timestamps
- Chat history limit (50 messages) and download option
- Predefined responses for simple inputs

## Prerequisites
- Python 3.8+
- Git (for cloning the repository)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/himu0023/RAGChat
   cd local-rag-chatbot

2. Create a virtual environment (optional but recommended):
   - python -m venv venv
   - source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
  - pip install streamlit langchain langchain-community langchain-huggingface transformers PyPDF2 faiss-cpu pytz sentence_transformers pypdf

4. Run the app:
   - streamlit run start.py


## Usage:

- Upload one or more PDFs via the sidebar.
- Type queries in the chat input; use simple phrases (e.g., "thank you") or document-related questions.
- Clear chat or download history as needed.


## Contributing
- Feel free to fork, improve, and submit pull requests!


## Credits

- Built with ❤️ by Himanshu Bisht
- Powered by Hugging Face and Streamlit
