import base64
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from dotenv import load_dotenv
from tools.gpu_config import GPU_CONFIG

load_dotenv()

VISION_REPO_ID = os.getenv("VISION_REPO_ID", "mys/ggml_llava-v1.5-7b")
VISION_MODEL_FILE = os.getenv("VISION_MODEL_FILE", "ggml-model-q4_k.gguf")
VISION_MMPROJ_FILE = os.getenv("VISION_MMPROJ_FILE", "mmproj-model-f16.gguf")
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))

_vision_llm = None

def get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Ensuring vision models are downloaded to {MODEL_DIR}...")
        model_path = hf_hub_download(
            repo_id=VISION_REPO_ID,
            filename=VISION_MODEL_FILE,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        mmproj_path = hf_hub_download(
            repo_id=VISION_REPO_ID,
            filename=VISION_MMPROJ_FILE,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        
        n_gpu_layers = GPU_CONFIG["n_gpu_layers"]
        mode = "GPU" if GPU_CONFIG["use_gpu"] else "CPU"
        print(f"Loading local Vision model ({mode} mode, {n_gpu_layers} GPU layers)...")
        
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        _vision_llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=n_gpu_layers,  # Dynamic: -1 (all GPU), 0 (CPU), or partial
            logits_all=True,
            verbose=False
        )
    return _vision_llm

def analyze_image(file_path, prompt="Describe this image in detail and identify any objects or text."):
    """
    Sends the image to a local Vision model via llama-cpp-python to generate captions.
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        vision_llm = get_vision_llm()
        
        messages = [
            {"role": "system", "content": "You are a helpful visual assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        response = vision_llm.create_chat_completion(
            messages=messages,
            stream=False
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Vision processing error: {str(e)}"
