import json
import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate").replace("generate", "chat")

def query_llm(messages, model=None):
    """
    Communicates with the local LLM via Ollama's Chat API.
    `messages` should be a list of dicts: {"role": "system|user|assistant", "content": "..."}
    """
    target_model = model if model else LLM_MODEL
    url = OLLAMA_API_URL
    
    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False
    }
    
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"LLM Connection Error: {str(e)}\nMake sure Ollama is running and '{target_model}' is pulled."
