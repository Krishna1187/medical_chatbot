import os

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ==========================
# Streamlit Page Settings
# ==========================

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical RAG Chatbot")
st.write("Ask any question from your Medical PDF.")

# ==========================
# Load LLM
# ==========================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ==========================
# Load Embeddings
# ==========================

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ==========================
# Load FAISS Database
# ==========================

@st.cache_resource
def load_db():

    embedding = load_embedding()

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding,
        allow_dangerous_deserialization=True
    )

    return db

llm = load_llm()
db = load_db()

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# ==========================
# Chat History
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================
# User Input
# ==========================

query = st.chat_input("Ask your medical question...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching Medical Knowledge..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a professional medical assistant.

Answer ONLY from the supplied medical documents.

If the answer is not available, reply:

"I couldn't find this information in the provided medical documents."

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        answer = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("📚 Source Documents"):

            for i, doc in enumerate(docs, 1):

                st.write(f"### Source {i}")

                st.write(doc.page_content[:700] + "...")

                st.write(doc.metadata)