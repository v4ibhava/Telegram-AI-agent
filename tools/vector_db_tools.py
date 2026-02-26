import chromadb
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
# Store DB relative to the main project directory, not inside the tools folder
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), CHROMA_DB_PATH)

try:
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name="agent_memory")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    collection = None

def store_in_memory(text, embedding, metadata=None):
    """Store a chunk of text, its embedding, and metadata into the vector DB. (CREATE)"""
    if not collection or not text.strip() or not embedding:
        return None
    
    doc_id = str(uuid.uuid4())
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}],
        ids=[doc_id]
    )
    return doc_id

def retrieve_from_memory(query_embedding, n_results=3):
    """Retrieve top N matching documents for a given query embedding. (READ/RAG)"""
    if not collection or not query_embedding:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Return matched documents
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return []

def delete_by_source(source_name):
    """Removes all embedded chunks associated with a specific file source. (DELETE)"""
    if not collection: return False
    collection.delete(where={"source": source_name})
    return True

def wipe_all_memory():
    """Completely wipes the entire vector memory database. (NUCLEAR DELETE)"""
    global collection
    try:
        if collection:
            count = collection.count()
            chroma_client.delete_collection(name="agent_memory")
            collection = chroma_client.get_or_create_collection(name="agent_memory")
            return count
        return 0
    except Exception as e:
        print(f"Error wiping memory: {e}")
        return 0
