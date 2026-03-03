import os
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_FOLDER = "pdfs"
INDEX_PATH = "index.faiss"
META_PATH = "meta.pkl"


def main():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

    documents = []

    for pdf_file in os.listdir(PDF_FOLDER):

        if not pdf_file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        reader = PdfReader(pdf_path)

        print(f"Ingesting {pdf_file}")

        for page_no, page in enumerate(reader.pages):

            text = page.extract_text()

            if not text:
                continue

            chunks = splitter.split_text(text)

            for idx, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "page": page_no + 1,
                    "book": pdf_file,
                    "chunk_id": f"{pdf_file}_{page_no}_{idx}"
                })

    texts = [d["text"] for d in documents]

    embeddings = embedder.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("Multi-PDF ingestion complete.")


if __name__ == "__main__":
    main()
