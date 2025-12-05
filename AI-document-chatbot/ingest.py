import os 
import argparse 
import pickle 
from typing import List

import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

# api = os.getenv("OPENAI_API_KEY")
# print(api)


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"


def read_pdfs_from_dir(pdf_dir: str) -> List[dict]:
    documents = []
    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, filename)
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            try:
                full_text += page.extract_text() or ""
                full_text += "\n"
            except Exception:
                continue
        documents.append({"filename": filename, "text": full_text})
    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.replace("\r", " ").replace("\n", " ")
    tokens = text.split(" ")
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main(pdf_dir: str):
    print(f"Reading PDFs from: {pdf_dir}")
    docs = read_pdfs_from_dir(pdf_dir)
    if not docs:
        print("No PDF files found.")
        return

    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "filename": doc["filename"],
                    "chunk_index": i,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    print("Creating embeddings...")
    embeddings = []
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        emb = embed_texts(batch)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    index_path = os.path.join(DATA_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    doc_path = os.path.join(DATA_DIR, "documents.pkl")
    with open(doc_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": metadata}, f)
    print(f"Documents metadata saved to {doc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF files")
    args = parser.parse_args()
    main(args.pdf_dir)

