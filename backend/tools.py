#backend/tools.py
# Step1: Setup Ollama with Medgemma tool
import ollama
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def query_medgemma(prompt: str) -> str:
    """
    Calls MedGemma model with a therapist personality profile.
    Returns responses as an empathic mental health professional.
    """
    system_prompt = """You are Dr. Emily Hartman, a warm and experienced clinical psychologist. 
    Respond to patients with:

    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")

    Key principles:
    - Never use brackets or labels
    - Blend elements seamlessly
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem
    """
    
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                'num_predict': 350,  # Slightly higher for structured responses
                'temperature': 0.7,  # Balanced creativity/accuracy
                'top_p': 0.9        # For diverse but relevant responses
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"I'm having technical difficulties, but I want you to know your feelings matter. Please try again shortly."

#print(query_medgemma("I feel so anxious all the time and can't focus on anything."))

# Step2: Setup Twilio calling API tool
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, EMERGENCY_CONTACT

def call_emergency():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=EMERGENCY_CONTACT,
        from_=TWILIO_FROM_NUMBER,
        url="http://demo.twilio.com/docs/voice.xml"  # Can customize message
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
EMB_FILE = os.path.join(EMB_DIR, "exam_texts.npy")
META_FILE = os.path.join(EMB_DIR, "exam_meta.json")
INDEX_FILE = os.path.join(EMB_DIR, "exam_faiss.index")

_embedding_model = None
_faiss_index = None
_meta = None


def _ensure_index_loaded():
    global _embedding_model, _faiss_index, _meta

    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if _faiss_index is None:
        _faiss_index = faiss.read_index(INDEX_FILE)

    if _meta is None:
        with open(META_FILE, "r", encoding="utf-8") as f:
            _meta = json.load(f)


def retrieve_exam_contexts(query: str, k: int = 5):
    _ensure_index_loaded()
    q_emb = _embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = _faiss_index.search(q_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        item = _meta[idx]
        results.append({
            "score": float(score),
            "instruction": item["instruction"],
            "response": item["response"]
        })
    return results


def ask_exam_rag(query: str) -> str:
    """
    Retrieves the most relevant exam-stress contexts and sends them to MedGemma for grounded response.
    """
    contexts = retrieve_exam_contexts(query, k=5)

    context_text = "\n\n".join(
        [f"Context {i+1} (score={c['score']:.3f}):\nQ: {c['instruction']}\nA: {c['response']}"
         for i, c in enumerate(contexts)]
    )

    system_prompt = """You are Dr. Emily Hartman, a warm clinical psychologist.
Use the retrieved exam-related contexts to give grounded, supportive, and empathetic guidance.
Do not mention the retrieval system. Just incorporate the ideas naturally.
Always ask an open-ended follow-up question.
"""

    user_msg = (
        f"Retrieved Contexts:\n{context_text}\n\n"
        f"User Question: {query}\n\n"
        f"Respond using the context when relevant."
    )

    response = ollama.chat(
        model="alibayram/medgemma:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        options={"temperature": 0.6, "num_predict": 250}
    )

    return response["message"]["content"]
