
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import PyPDF2
import streamlit as st


# STEP 1: Load Gemini API key

load_dotenv()
genai.configure(api_key=".......")
print("Configured API key successfully!")

# STEP 2: Extract text from PDF

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# STEP 3: Create FAISS vector store

def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# STEP 4: Gemini-based query

def answer_query(query, retriever):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    You are an intelligent assistant. Use the context below to answer the user's question accurately.

    Context:
    {context}

    Question:
    {query}
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# STEP 5: Streamlit UI

st.title("📘 PDF-based RAG Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully! Creating knowledge base...")
    text = extract_text_from_pdf(uploaded_file)
    vectorstore = create_vectorstore(text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.success("Knowledge base ready! Ask questions below:")

    # Chat loop
    query = st.text_input("Ask a question:")
    if query:
        if query.lower() in ["quit", "exit"]:
            st.write("👋 Exiting chat.")
        else:
            answer = answer_query(query, retriever)
            st.write("🤖 Gemini:", answer)
