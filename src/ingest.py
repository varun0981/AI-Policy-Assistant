from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --------------------------------
# LOAD PDF FILES
# --------------------------------
documents = []

folder_path = "data/policies"

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        file_path = os.path.join(
            folder_path,
            file
        )

        print(f"Loading: {file}")

        loader = PyPDFLoader(
            file_path
        )

        docs = loader.load()

        documents.extend(docs)

# --------------------------------
# CHUNK DOCUMENTS
# --------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

print(f"Total Pages: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")

# --------------------------------
# EMBEDDING MODEL
# --------------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------
# CHROMADB
# --------------------------------
client = chromadb.PersistentClient(
    path="vectordb"
)

# Delete old collection if exists
try:
    client.delete_collection(
        "company_policies"
    )
    print("Old collection deleted.")
except:
    pass

collection = client.get_or_create_collection(
    name="company_policies"
)

# --------------------------------
# STORE CHUNKS
# --------------------------------
for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk.page_content
    ).tolist()

    source_file = chunk.metadata.get(
        "source",
        "Unknown PDF"
    )

    collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[chunk.page_content],
    metadatas=[{
        "source": chunk.metadata.get("source", "Unknown"),
        "page": chunk.metadata.get("page", 0)
    }]
)

print("Embeddings stored successfully!")
print("Vector Database Ready!")