import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from deep_translator import GoogleTranslator


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Police Legal Assistant",
    page_icon="👮",
    layout="wide"
)

# ==========================================
# LOAD CSS
# ==========================================

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("👮 Police Legal Assistant")

language = st.sidebar.radio(
    "Choose Language",
    ["English", "Tamil"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Case Categories")

st.sidebar.button("Theft")
st.sidebar.button("Murder")
st.sidebar.button("Cyber Crime")
st.sidebar.button("Domestic Violence")
st.sidebar.button("Accident")

st.sidebar.markdown("---")


# ==========================================
# TITLE
# ==========================================

st.title("⚖️ Police Law RAG Bot")

st.write(
    "Ask legal questions based on uploaded police manuals and law PDFs."
)


# ==========================================
# LOAD PDF DOCUMENTS
# ==========================================

@st.cache_resource
def load_documents():

    documents = []

    folder_path = "data/laws"

    if not os.path.exists(folder_path):
        st.error("laws folder not found")
        return []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    return documents


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

@st.cache_resource
def create_vector_store():

    documents = load_documents()

    if len(documents) == 0:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embedding=embeddings
    )

    return vectorstore


vectorstore = create_vector_store()


# ==========================================
# LOAD AI MODEL
# ==========================================

@st.cache_resource
def load_llm():

    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    return generator


llm = load_llm()


# ==========================================
# CHECK VECTOR DATABASE
# ==========================================

if vectorstore is None:

    st.warning("Please add PDF files inside data/laws folder")

    st.stop()


# ==========================================
# USER INPUT
# ==========================================

user_question = st.chat_input(
    "Ask your legal question..."
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if user_question:

    with st.chat_message("user"):
        st.write(user_question)

    # ======================================
    # RETRIEVE DOCUMENTS
    # ======================================

    docs = vectorstore.similarity_search(
        user_question,
        k=3
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    # ======================================
    # IMPROVED PROMPT
    # ======================================

    prompt = f"""
    You are a Police Legal Assistant.

    Use ONLY the provided legal context.

    User Question:
    {user_question}

    Legal Context:
    {context}

    Give answer in this EXACT format:

    1. Case Type
    2. Relevant Law Section
    3. Explanation
    4. Punishment
    5. FIR Guidance
    6. Important Notes

    Give answer step-by-step in points.
    Do NOT give paragraph answers.
    """

    # ======================================
    # GENERATE RESPONSE
    # ======================================

    result = llm(prompt)

    final_response = result[0]["generated_text"]

    # ======================================
    # TAMIL TRANSLATION
    # ======================================

    if language == "Tamil":

        final_response = GoogleTranslator(
            source='auto',
            target='ta'
        ).translate(final_response)

    # ======================================
    # DISPLAY RESPONSE
    # ======================================

    with st.chat_message("assistant"):

        st.markdown(
            f"""
            <div class="chat-box">
            {final_response}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==================================
        # SOURCES
        # ==================================

        st.markdown("### 📚 Sources")

        for doc in docs:

            source = doc.metadata.get("source", "Unknown")

            page = doc.metadata.get("page", "N/A")

            st.write(f"📄 {source} | Page {page}")

