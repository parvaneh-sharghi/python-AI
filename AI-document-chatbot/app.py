# app.py
import os
import pickle
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DOCS_PATH = os.path.join(DATA_DIR, "documents.pkl")

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
    raise RuntimeError("Index or documents not found. Run ingest.py first.")

index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "rb") as f:
    data = pickle.load(f)

CHUNKS: List[str] = data["chunks"]
METADATA: List[dict] = data["metadata"]

app = FastAPI(title="AI Document Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    emb = response.data[0].embedding
    return np.array([emb], dtype="float32")


def search_similar_chunks(query: str, top_k: int = 4):
    query_emb = embed_query(query)
    distances, indices = index.search(query_emb, top_k)
    indices = indices[0]
    results = []
    for idx in indices:
        if idx < 0 or idx >= len(CHUNKS):
            continue
        results.append(
            {
                "text": CHUNKS[idx],
                "metadata": METADATA[idx],
            }
        )
    return results


def build_prompt(question: str, contexts: List[dict]) -> str:
    context_texts = "\n\n---\n\n".join(
        [f"[From {c['metadata']['filename']}]:\n{c['text']}" for c in contexts]
    )
    system_prompt = (
        "You are an AI assistant that answers questions only based on the provided document excerpts. "
        "If the answer is not clearly in the documents, say you don't know and do not hallucinate."
    )
    user_prompt = f"Question: {question}\n\nRelevant document excerpts:\n{context_texts}"
    return system_prompt, user_prompt


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    contexts = search_similar_chunks(request.question, top_k=request.top_k)
    if not contexts:
        return ChatResponse(
            answer="I couldn't find relevant information in the indexed documents.",
            sources=[],
        )

    system_prompt, user_prompt = build_prompt(request.question, contexts)

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content

    sources = [
        {
            "filename": c["metadata"]["filename"],
            "chunk_index": c["metadata"]["chunk_index"],
        }
        for c in contexts
    ]

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health_check():
    return {"status": "ok"}
