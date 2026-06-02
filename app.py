import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
st.set_page_config(
    page_title="AI Policy Assistant",
    page_icon="🤖",
    layout="wide"
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("AI Policy Assistant")
    st.write("Ask questions about company policies.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# PASTE HERE 👇
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1e3a8a,
        #2563eb
    );
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main page */
.main {
    background-color: #f8fafc;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 15px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #2563eb;
    border-radius: 15px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)
        

# -------------------------------
# API KEY
# -------------------------------
load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# -------------------------------
# GEMINI
# -------------------------------
model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

# -------------------------------
# EMBEDDING MODEL
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# -------------------------------
# CHROMADB
# -------------------------------
client = chromadb.PersistentClient(
    path="vectordb"
)

collection = client.get_or_create_collection(
    name="company_policies"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("AI Company Policy Assistant")

# -------------------------------
# PDF UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs(
        "uploaded_pdfs",
        exist_ok=True
    )

    all_documents = []

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            "uploaded_pdfs",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )

        loader = PyPDFLoader(
            file_path
        )

        docs = loader.load()

        all_documents.extend(docs)

    # ---------------------------
    # CHUNKING
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        all_documents
    )

    # ---------------------------
    # CLEAR OLD DATA
    # ---------------------------
    try:
        client.delete_collection(
            "company_policies"
        )
    except:
        pass

    collection = client.get_or_create_collection(
        name="company_policies"
    )

    # ---------------------------
    # STORE EMBEDDINGS
    # ---------------------------
    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk.page_content
        ).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[
                chunk.page_content
            ],
            metadatas=[
                {
                    "source": chunk.metadata.get("source", "Unknown PDF"),
                    "page": chunk.metadata.get("page", 0)
                }
            ]
        )

    st.success(
        f"{len(chunks)} chunks stored in ChromaDB!"
    )

# -------------------------------
# QUESTION BOX
# -------------------------------
question = st.chat_input(
    "Ask a policy question..."
)

# -------------------------------
# QUERY
# -------------------------------
if question:

    with st.chat_message("user"):
        st.write(question)

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    # generate answer
    prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{question}
"""
    response = model.generate_content(prompt)
    with st.chat_message("assistant"):
        st.write(response.text)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": response.text
        }
    )

    st.subheader("Source")

    metadata = results["metadatas"][0][0]

    st.info(
        f"""
📄 File: {os.path.basename(metadata['source'])}

📃 Page: {metadata['page'] + 1}
"""
    )

    with st.expander("View Source Text"):
        st.write(results["documents"][0][0])

    st.subheader("Chat History")

for chat in reversed(st.session_state.chat_history):

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])