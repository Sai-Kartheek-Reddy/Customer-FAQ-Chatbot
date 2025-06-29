import json
import os

import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

# ------------------------------------------
# Setup
# ------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# ------------------------------------------
# Load Original JSONL
# ------------------------------------------
input_file = "Jup-RAG-Doc.jsonl"
output_file = "Jup-RAG-Chunked.jsonl"

raw_docs = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        raw_docs.append(
            Document(
                page_content=item["content"],
                metadata={
                    "category": item["category"],
                    "title": item["title"],
                    "chunk_id": item.get("id", "")
                }
            )
        )

# print(f"✅ Loaded {len(raw_docs)} raw documents.")


# ------------------------------------------
# Chunking Setup
# ------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,         # You can tune between 500-1000
    chunk_overlap=100       # Helps preserve context flow
)

chunked_docs = text_splitter.split_documents(raw_docs)

# print(f"✅ Generated {len(chunked_docs)} chunks after splitting.")


# ------------------------------------------
# Save to JSONL
# ------------------------------------------
with open(output_file, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(chunked_docs):
        json.dump({
            "content": doc.page_content,
            "category": doc.metadata.get("category", ""),
            "title": doc.metadata.get("title", ""),
            "chunk_id": doc.metadata.get("chunk_id", "") + f"_chunk_{idx}"
        }, f, ensure_ascii=False)
        f.write("\n")

# print(f"✅ Saved chunked documents to {output_file}")


# Load RAG Documents
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
                    "chunk_id": item.get("id", "")
                }
            )
        )
# print(f"✅ Loaded {len(documents)} documents.")


# ------------------------------------------
# Embeddings + FAISS
# ------------------------------------------
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

faiss_db = FAISS.from_documents(documents, embeddings)
faiss_db.save_local("faiss_index")
faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# print("✅ FAISS index loaded successfully.")


# ------------------------------------------
# Load Llama Model
# ------------------------------------------
model_id = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
#     quantization_config=bnb_config
)

# print("✅ Llama model loaded successfully.")


# ------------------------------------------
# Helper: Check if Query is Relevant to Jupiter
# ------------------------------------------
def is_query_about_jupiter(user_query):
    instruction = f"""
Determine whether the user query is related to Jupiter Money (a financial company offering products like savings, investments, loans, credit cards, and customer services).

If the query is about Jupiter's products, services, policies, customer care, company information, or anything directly related to Jupiter, reply with 'Yes'.

If the query is not related to Jupiter (like questions about geography, weather, history, etc.), reply with 'No'.

User Query: {user_query}
Answer:"""

    inputs = tokenizer(instruction, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True).lower()

    return "yes" in response


# ------------------------------------------
# Prompt Builder
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
# RAG Query Function
# ------------------------------------------
def rag_query(user_query, top_k=5):
    # First, check if question is about Jupiter
    if not is_query_about_jupiter(user_query):
        return "I am an assistant specifically for Jupiter Money. Please ask questions related to Jupiter's products, services, or company. I'm happy to clarify your queries."

    # Semantic Retrieval
    retrieved_docs = faiss_db.similarity_search(user_query, k=top_k)

    if not retrieved_docs:
        return "I'm not aware of this information based on the available data."

    # Prompt
    prompt = generate_prompt(user_query, retrieved_docs)

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    output = model.generate(
        **input_ids,
        max_new_tokens=512,
        num_beams=4,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )


    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return response


# ------------------------------------------
# Run Loop
# ------------------------------------------
if __name__ == "__main__":
    print("\n🔷 Welcome to Jupiter Assistant! 🔷")
    print("Ask me anything related to Jupiter Money.\n")

    while True:
        query = input("Enter your question (or 'exit'): ")
        if query.lower() == "exit":
            print("\nNice talking to you. \nBest regards, \nJupiter Assistant.\n")
            break

        result = rag_query(query)
        print("\n>>", result)
