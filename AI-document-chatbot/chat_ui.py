# chat_ui.py
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{API_BASE}/chat"
HEALTH_ENDPOINT = f"{API_BASE}/health"

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="centered")
st.title("📄 AI Document Chatbot")
st.caption("Ask questions about your indexed PDF documents (RAG + FAISS + FastAPI).")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Context chunks (top_k)", 1, 10, 4)
    st.write("API status:")

    # Check API health
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        if r.status_code == 200:
            st.success("API is running ✅")
        else:
            st.error(f"API error: {r.status_code}")
    except Exception:
        st.error("API not reachable ❌")
        st.markdown("Run in another terminal:")
        st.code("uvicorn app:app --reload", language="bash")

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
question = st.chat_input("Type your question…")

if question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # call API
    payload = {"question": question, "top_k": top_k}

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
                if resp.status_code != 200:
                    st.error(f"API error: {resp.status_code}\n\n{resp.text}")
                    answer = "Sorry — the API returned an error."
                    sources = []
                else:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    st.markdown(answer)

                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources:
                            st.markdown(f"- `{s['filename']}` (chunk {s['chunk_index']})")
            except Exception as e:
                st.error(f"Failed to call API: {e}")
                answer = "Sorry — I couldn't reach the API."
                sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer})
