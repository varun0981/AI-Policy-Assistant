from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="vectordb"
)

collection = client.get_collection(
    "company_policies"
)

# User question
query = "How many casual leaves do employees get?"

# Convert question to embedding
query_embedding = model.encode(
    query
).tolist()

# Search database
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nQUESTION:")
print(query)

print("\nTOP MATCHES:\n")

for doc in results["documents"][0]:
    print(doc)
    print("\n-----------------\n")