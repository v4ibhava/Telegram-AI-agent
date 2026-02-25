import base64
import json
import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL = os.getenv("VISION_MODEL", "llava")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

def analyze_image(file_path, prompt="Describe this image in detail and identify any objects or text."):
    """
    Sends the image to a local Vision model (e.g., LLaVA or Phi-3-vision running via Ollama)
    to generate captions and analyze contents.
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
        payload = {
            "model": VISION_MODEL, 
            "prompt": prompt,
            "images": [encoded_string],
            "stream": False
        }
        
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(OLLAMA_API_URL, data=json.dumps(payload).encode('utf-8'), headers=headers)
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except Exception as e:
        return f"Vision processing error: {str(e)}"
