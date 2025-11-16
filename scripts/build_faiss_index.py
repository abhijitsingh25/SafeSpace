import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import faiss

# ABSOLUTE PATH – FIXED
DATA_PATH = r"C:\GenAi\SafeSpace\data\mental_health_exams.jsonl"

# Embedding directory (relative to project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

EMB_FILE = os.path.join(EMB_DIR, "exam_texts.npy")
META_FILE = os.path.join(EMB_DIR, "exam_meta.json")
INDEX_FILE = os.path.join(EMB_DIR, "exam_faiss.index")

# -------- LOAD DATASET SAFELY ----------
texts = []
meta = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()

        # skip empty lines
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"⚠️ Skipping bad line {i+1}: {line}")
            continue

        instruct = obj.get("instruction", "").strip()
        resp = obj.get("response", "").strip()

        combined = f"Q: {instruct}\nA: {resp}"
        texts.append(combined)
        meta.append({"id": i, "instruction": instruct, "response": resp})

print(f"Loaded {len(texts)} valid entries.")

# -------- CREATE EMBEDDINGS ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save(EMB_FILE, embeddings)
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# -------- BUILD FAISS INDEX ----------
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

print("🎉 Successfully created embeddings + FAISS index!")
print("Saved in:", EMB_DIR)
