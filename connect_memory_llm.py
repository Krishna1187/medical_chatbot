import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY")
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore/db_faiss",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k":3})

query = input("Write Query Here : ")

docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are an expert medical assistant.

Use ONLY the following context to answer the user's question.

If the answer is not present in the context, reply:
"I couldn't find this information in the provided medical documents."

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)

print("\nSource Documents:\n")

for doc in docs:
    print(doc.metadata)