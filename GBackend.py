import json
import os
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai


# ------------------------------------------
# ✅ Configure Gemini API (from secrets)
# ------------------------------------------
import streamlit as st
genai.configure(api_key=st.secrets["GENAI_API_KEY"])


# ------------------------------------------
# ✅ Load Pre-Chunked Documents
# ------------------------------------------
jsonl_path = "Jup-RAG-Chunked.jsonl"
documents = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        documents.append(
            Document(
                page_content=item["content"],
                metadata={
                    "category": item["category"],
                    "title": item["title"],
                    "chunk_id": item.get("chunk_id", "")
                }
            )
        )


# ------------------------------------------
# ✅ Load FAISS Index
# ------------------------------------------
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

faiss_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ------------------------------------------
# ✅ Load Gemini Model
# ------------------------------------------
model = genai.GenerativeModel('gemini-1.5-flash')


# ------------------------------------------
# ✅ Helper: Check if Query is About Jupiter
# ------------------------------------------
def is_query_about_jupiter(user_query):
    instruction = f"""
Determine whether the user query is related to Jupiter Money (a financial company offering products like savings, investments, loans, credit cards, and customer services).

If the query is about Jupiter's products, services, policies, customer care, company information, or anything directly related to Jupiter, reply with 'Yes'.

If the query is not related to Jupiter (like questions about geography, weather, history, etc.), reply with 'No'.

User Query: {user_query}
Answer:
"""
    response = model.generate_content(instruction)
    result = response.text.strip().lower()

    return "yes" in result


# ------------------------------------------
# ✅ Prompt Builder
# ------------------------------------------
def generate_prompt(user_query, context_chunks):
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])

    prompt = f"""
You are a financial assistant for Jupiter Money.

------------------
Context:
{context_text}
------------------

Instructions:
- Use only the provided context to answer.
- If the answer is clearly not present in the context, reply: "I'm not aware of this information based on the available data."
- Avoid hallucinating or making up answers.
- Be precise, concise, and helpful.
- Provide a complete answer without stopping abruptly or mid-sentence.

User Query: {user_query}

Your Answer:
"""
    return prompt.strip()


# ------------------------------------------
# ✅ RAG Query Function
# ------------------------------------------
def rag_query(user_query, top_k=5):
    if not is_query_about_jupiter(user_query):
        return (
            "I am an assistant specifically for Jupiter Money. "
            "Please ask questions related to Jupiter's products, services, or company. "
            "I'm happy to help clarify your queries."
        )

    retrieved_docs = faiss_db.similarity_search(user_query, k=top_k)

    if not retrieved_docs:
        return "I'm not aware of this information based on the available data."

    prompt = generate_prompt(user_query, retrieved_docs)

    response = model.generate_content(prompt)
    result = response.text.strip()

    return result
