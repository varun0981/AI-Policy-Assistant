from langchain_community.document_loaders import PyPDFLoader
import os

documents = []

folder_path = "data/policies"

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        print(f"Loading {file}")

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)

print("\nTotal Pages Loaded:", len(documents))

print("\nFirst Page Preview:\n")

print(documents[0].page_content[:500])