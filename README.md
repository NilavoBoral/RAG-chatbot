# RAG-chatbot
A Retrieval-Augmented Generation (RAG) based chatbot that answers questions using information from your PDF documents and displays the source of the response (PDF name and page number).


---

## Setup

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Environment Variable

Create a `.env` file in the root directory:

```
# Cloud LLM API Key
GEMINI_API_KEY=your_gemini_api_key # Provide your cloud provider's API key
```

---

## Steps to Run

### 1️⃣ Ingest Documents

Create a folder named `pdfs` in the root directory (if it does not exist). Place all PDFs inside the `pdfs/` folder, then run:

```bash
python 1_ingest.py
```

This will:
- Load all PDFs
- Split text into chunks
- Generate embeddings
- Store them in the vector database


### 2️⃣ Run Chatbot UI

```bash
streamlit run 2_streamlit_app.py
```

This will:
- Launch the Streamlit chatbot interface
- Internally use `rag.py` for retrieval and response generation
- Display the source of each answer (PDF name and page number)

---

## Notes

- Make sure ingestion is completed before running the chatbot.
- Add new PDFs to the `pdfs/` folder and rerun `1_ingest.py` to update embeddings.
- Embeddings are currently stored locally in the root directory. The storage layer can be replaced with any vector database (FAISS, Chroma, Pinecone, etc.).
- The project currently uses Gemini via `GEMINI_API_KEY`. The cloud model provider can be changed as needed.

---
