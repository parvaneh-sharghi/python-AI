import os
import argparse
import pickle
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ----------------- Paths & constants -----------------

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DOCS_PATH = os.path.join(DATA_DIR, "documents.pkl")          # stores chunks + metadata
REGISTRY_PATH = os.path.join(DATA_DIR, "file_registry.json") # tracks processed PDFs

EMBEDDING_MODEL = "text-embedding-3-small"

# NOTE:
# In your original code you chunk by "words", so CHUNK_SIZE/OVERLAP represent word counts.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

print("INGEST VERSION WITH REGISTRY ✅")
print("REGISTRY_PATH =", REGISTRY_PATH)

# ----------------- OpenAI client -----------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- Utility functions -----------------

def read_pdfs_from_dir(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Read all PDFs from a directory and extract:
      - filename
      - path
      - full text (best effort)
      - mtime (modification time)
    """
    documents: List[Dict[str, Any]] = []
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

        stat = os.stat(path)
        documents.append(
            {
                "filename": filename,
                "path": path,
                "text": full_text,
                "mtime": stat.st_mtime,
            }
        )
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    """
    text = text.replace("\r", " ").replace("\n", " ")
    words = text.split()

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks using OpenAI embeddings API.
    """
    if not texts:
        # If empty, return empty (0, dim) array; dim will be resolved by caller if needed.
        return np.zeros((0, 0), dtype="float32")

    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
    return np.vstack(vectors)


def load_existing_index() -> Tuple[Any, Any]:
    """
    Load existing FAISS index and docs data if available.
    Returns (index, docs_data) or (None, None)
    """
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs_data = pickle.load(f)

    return index, docs_data


def load_registry() -> Dict[str, Any]:
    """
    Load registry JSON which tracks processed PDFs.
    Format: { "file.pdf": {"mtime": 123456.789}, ... }
    """
    if not os.path.exists(REGISTRY_PATH):
        return {}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any]) -> None:
    """
    Persist registry to disk.
    """
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


# ----------------- Core ingest logic -----------------

def full_rebuild(pdf_dir: str) -> None:
    """
    Rebuild the FAISS index from scratch for all PDFs in pdf_dir.
    Creates/overwrites:
      - index.faiss
      - documents.pkl
      - file_registry.json
    """
    print("⚠ Full rebuild from scratch...")

    docs = read_pdfs_from_dir(pdf_dir)

    all_chunks: List[str] = []
    all_metadata: List[dict] = []

    for doc in docs:
        doc_chunks = chunk_text(doc["text"])
        for i, ch in enumerate(doc_chunks):
            all_chunks.append(ch)
            all_metadata.append(
                {
                    "filename": doc["filename"],
                    "chunk_index": i,
                }
            )

    if not all_chunks:
        raise RuntimeError("No extractable text found in PDFs (nothing to index).")

    embeddings = embed_texts(all_chunks)
    if embeddings.size == 0:
        raise RuntimeError("Embedding returned empty result. Check your API key / model / inputs.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    # Registry tracks each file and its last modified time
    registry = {doc["filename"]: {"mtime": doc["mtime"]} for doc in docs}
    save_registry(registry)

    print(f"✅ Full rebuild done. PDFs: {len(docs)} | chunks: {len(all_chunks)}")
    print(f"Saved: {INDEX_PATH}, {DOCS_PATH}, {REGISTRY_PATH}")


def incremental_update(pdf_dir: str) -> None:
    """
    Incrementally add only NEW PDFs.
    If any existing PDF is modified, do a full rebuild (safe + simple).
    """
    index, docs_data = load_existing_index()
    registry = load_registry()
    docs = read_pdfs_from_dir(pdf_dir)

    new_files: List[Dict[str, Any]] = []
    changed_files: List[Dict[str, Any]] = []

    for doc in docs:
        name, mtime = doc["filename"], doc["mtime"]
        if name not in registry:
            new_files.append(doc)
        else:
            # modified?
            if abs(registry[name]["mtime"] - mtime) > 1e-6:
                changed_files.append(doc)

    if changed_files:
        print("⚠ Some PDFs were modified since last ingest:")
        for d in changed_files:
            print(" -", d["filename"])
        print("Doing full rebuild (safe path).")
        full_rebuild(pdf_dir)
        return

    if not new_files:
        print("No new PDFs detected. Nothing to do.")
        return

    if index is None or docs_data is None:
        print("No existing index found. Doing full rebuild.")
        full_rebuild(pdf_dir)
        return

    print(f"➕ Adding {len(new_files)} new PDF(s) incrementally...")

    chunks: List[str] = docs_data["chunks"]
    metadata: List[dict] = docs_data["metadata"]

    new_chunks: List[str] = []
    new_metadata: List[dict] = []

    for doc in new_files:
        doc_chunks = chunk_text(doc["text"])
        for i, ch in enumerate(doc_chunks):
            new_chunks.append(ch)
            new_metadata.append(
                {
                    "filename": doc["filename"],
                    "chunk_index": i,
                }
            )

    if not new_chunks:
        print("New PDFs had no extractable text. Nothing added.")
        return

    new_embeddings = embed_texts(new_chunks)
    if new_embeddings.size == 0:
        raise RuntimeError("Embedding returned empty result for new chunks.")

    # Must match same dimension as existing index
    if new_embeddings.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimension mismatch. index.d={index.d} but new_embeddings={new_embeddings.shape[1]}"
        )

    index.add(new_embeddings)

    # Append to stored docs
    chunks.extend(new_chunks)
    metadata.extend(new_metadata)

    # Save updated data
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)

    # Update registry with new files
    for doc in new_files:
        registry[doc["filename"]] = {"mtime": doc["mtime"]}
    save_registry(registry)

    print(f"✅ Incremental update done. Added PDFs: {len(new_files)} | added chunks: {len(new_chunks)}")
    print(f"Updated: {INDEX_PATH}, {DOCS_PATH}, {REGISTRY_PATH}")


def main(pdf_dir: str) -> None:
    print(">>> ENTERED main()")
    print("PDF DIR =", pdf_dir)

    if not os.path.exists(pdf_dir):
        raise RuntimeError(f"PDF directory not found: {pdf_dir}")

    # If any of these is missing, do full rebuild
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH) and os.path.exists(REGISTRY_PATH)):
        print("No existing index/docs/registry found → full rebuild")
        full_rebuild(pdf_dir)
    else:
        incremental_update(pdf_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS index (incremental + registry).")
    parser.add_argument("--pdf-dir", type=str, default="./pdfs", help="Directory containing PDF files")
    args = parser.parse_args()

    main(args.pdf_dir)
