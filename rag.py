import os
import faiss
import pickle
import json
import uuid
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

INDEX_PATH = "index.faiss"
META_PATH = "meta.pkl"
CHAT_FILE = "chat.json"

SYSTEM_PROMPT = """
You are a PDF research assistant.

Rules:
- Use ONLY provided context and previous chat history.
- If the answer is missing, respond with: "No reliable source found."
- If the answer is NOT based on the provided context, set Sources to False. Otherwise, set Sources to True.
- If the question is general (e.g., "hi", "how are you", "good morning") and cannot be answered with the provided context and previous chat history, respond with a generic answer and set Sources to False.
- Sources must be STRICTLY either True or False, following the above rules.

Output format:

Answer:
...

Sources: True/False
"""


# ---------------- CACHE HEAVY OBJECTS ---------------- #

@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def load_index():
    return faiss.read_index(INDEX_PATH)

@lru_cache(maxsize=1)
def load_metadata():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


embedder = load_embedder()
index = load_index()
metadata = load_metadata()


# ---------------- SESSION STORAGE ---------------- #

def _load_all_sessions():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_all_sessions(data):
    with open(CHAT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def new_session():
    return str(uuid.uuid4())


# ---------------- RETRIEVER ---------------- #

def retrieve(query, k=6):

    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)

    results = []

    for dist, idx in zip(D[0], I[0]):
        doc = metadata[idx]

        results.append({
            "text": doc["text"],
            "page": doc["page"],
            "book": doc["book"],
            "score": float(dist)
        })

    return results


# ---------------- MAIN ASK ---------------- #

def ask(question, session_id):

    sessions = _load_all_sessions()

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    history.append({"role": "user", "content": question})

    history_text = ""
    for h in history:
        history_text += f"{h['role'].capitalize()}: {h['content']}\n"

    results = retrieve(question)

    context = "\n\n".join(
        [f"[{r['book']} Page {r['page']}] {r['text']}" for r in results]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Conversation so far:
{history_text}

Context:
{context}

Current Question:
{question}
"""

    try:
        response = model.generate_content(prompt)

        # print("MODEL RESPONSE:", response.text)

        text = response.text
        parts = text.split("Sources:")

        answer = parts[0].replace("Answer:", "").strip()
        source_flag = parts[1].strip().lower() if len(parts) > 1 else "false"

    except Exception:
        return {
            "answer": "API/token limit exceeded. Please start a new session.",
            "sources": []
        }

    # history.append({"role": "assistant", "content": answer})

    # sessions[session_id] = history
    # _save_all_sessions(sessions)

    sources = list(set([
        f"{r['book']} — Page {r['page']}" for r in results
    ]))

    if source_flag == "false":
        sources = []

    if sources:
        answer += "\n\nSources:\n"
        for s in sources:
            answer += f"- {s}\n"

    history.append({"role": "assistant", "content": answer})
    sessions[session_id] = history
    _save_all_sessions(sessions)

    return {
        "answer": answer,
        # "sources": sources
    }