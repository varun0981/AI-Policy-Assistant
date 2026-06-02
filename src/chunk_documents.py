from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

documents = []

folder_path = "data/policies"

# Load PDFs
for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            os.path.join(folder_path, file)
        )

        docs = loader.load()

        documents.extend(docs)

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Split documents
chunks = splitter.split_documents(
    documents
)

# Add source metadata
for chunk in chunks:

    chunk.metadata["source_file"] = (
        chunk.metadata.get(
            "source",
            "Unknown PDF"
        )
    )

print("Total Pages:", len(documents))
print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0].page_content)