import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load data

DATA_PATH ="data"
DB_FAISS_PATH = "vectorstore/db_faiss"
 
 #  Load PDF
def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls= PyPDFLoader,
        show_progress= True
    )
    documents = loader.load()


    print(f"\nLoaded{len(documents)} pages.")
    return documents 

# split  documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators= ["\n\n","\n", "."," ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# embedding model

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

# create faiss db

def create_vactor_db(chunks):
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(
        documents= chunks,
        embedding= embedding_model
    )
    Path("vectorstore").mkdir(exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print("\n FAISS Vector Database Cerated Successfully!")


# main _______
if __name__=="__main__":
    print("="*68)
    print("Medivcal RAG - Pdf Ingestion")
    print("="*60)
    docs = load_documents()
    chunks = split_documents(docs)
    create_vactor_db(chunks)
    print("\nDone!")