from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="vectordb"
)

collection = client.get_collection(
    "company_policies"
)

query = input("Ask a question: ")

query_embedding = embedding_model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n".join(
    results["documents"][0]
)

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

response = model.generate_content(
    prompt
)

print("\nANSWER:\n")
print(response.text)