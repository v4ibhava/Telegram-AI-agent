import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Load a CPU-friendly embedding model to save VRAM for the core LLM execution
print(f"Loading embedding model ({EMBEDDING_MODEL_NAME}) into CPU...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')

def get_embedding(text):
    """Generates an embedding vector for the provided text."""
    if not text or not text.strip():
        return []
    embeddings = model.encode([text])
    return embeddings[0].tolist()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits long text into manageable chunks before generating embeddings.
    Allows for overlapping windows to preserve context.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max(1, chunk_size - overlap)):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
