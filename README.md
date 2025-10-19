# 📄 RAG Chatbot with Google Gemini + FAISS


This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows you to ask questions about a PDF document. It uses:

- **Google Gemini API** for LLM-based answers  
- **FAISS** for vector similarity search  
- **HuggingFace embeddings** to convert PDF text chunks into vectors  

You can run it either as a **Jupyter/Colab Notebook** (`.ipynb`) for interactive use or a **Python script** (`.py`) for production.

---

## 🛠 Features

- Upload any PDF document  
- Split text into manageable chunks with overlap  
- Embed chunks into vectors using `all-MiniLM-L6-v2` model  
- Retrieve relevant chunks using **FAISS**  
- Ask questions and get **context-aware answers** from Gemini

---


```bash
pip install google-generativeai python-dotenv langchain PyPDF2 faiss-cpu sentence-transformers
```

## 🔑 API Setup

The project uses Google Gemini API. You need to set your API key.

##  🖥 Usage
1️⃣ Google Colab / Jupyter Notebook (.ipynb)

1. Open the notebook in Colab or Jupyter Notebook.

2. Upload your PDF (Colab file upload or local path in Jupyter).

3. Run all cells sequentially:

a. Imports & setup
b. API key configuration
c. PDF reading and vector store creation
d. Chat loop

4. Ask questions interactively using the input box.

2️⃣ Streamlit Web App (.py)

1. Make sure rag_streamlit.py is in your repo.

2. Ensure your API key is set in .env or in the code.

3. Install requirements if not done
4. Run the app:

streamlit run rag_streamlit.py

5. A browser window opens. You can upload a PDF and interact with the chatbot directly.


### 🔑 Notes

~ Notebook and Streamlit use the same backend logic, but Streamlit adds a web interface.

~ FAISS vector store is built in memory in both versions.

~ For multiple users or persistent usage, consider saving the FAISS index to disk

## References 
- https://faiss.ai/
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# 👩‍💻 Author

Divya – AI & ML Enthusiast | RAG & LLM Projects

