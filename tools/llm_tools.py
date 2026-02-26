import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from tools.gpu_config import GPU_CONFIG

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct-GGUF")
HF_FILENAME = os.getenv("HF_FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf")
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))

_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Ensuring model {HF_FILENAME} is downloaded to {MODEL_DIR}...")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        
        n_gpu_layers = GPU_CONFIG["n_gpu_layers"]
        mode = "GPU" if GPU_CONFIG["use_gpu"] else "CPU"
        print(f"Model downloaded/found. Initializing local LLaMA model ({mode} mode, {n_gpu_layers} GPU layers)...")
        
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=2048,        # Smaller context = faster inference
            n_threads=8,       # Threads for CPU-bound operations
            n_gpu_layers=n_gpu_layers,  # Dynamic: -1 (all GPU), 0 (CPU), or partial
            verbose=False
        )
    return _llm_instance

def query_llm(messages, model=None):
    """
    Communicates with the local LLM via llama-cpp-python.
    `messages` should be a list of dicts: {"role": "system|user|assistant", "content": "..."}
    """
    try:
        llm = get_llm()
        # Create chat completion
        response = llm.create_chat_completion(
            messages=messages,
            stream=False
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"LLM Connection Error: {str(e)}\nFailed to load or query the local model."
