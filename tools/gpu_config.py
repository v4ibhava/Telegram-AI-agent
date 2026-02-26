"""
GPU Configuration & Auto-Detection Module
Automatically detects NVIDIA GPU and configures optimal settings
for llama-cpp-python CUDA acceleration.
"""

import os
import subprocess
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# User override: "auto" (default), "true" (force GPU), "false" (force CPU)
USE_GPU = os.getenv("USE_GPU", "auto").lower().strip()


def detect_nvidia_gpu():
    """
    Detects NVIDIA GPU using nvidia-smi.
    Returns dict with GPU info or None if no GPU found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                return {
                    "name": parts[0].strip(),
                    "vram_mb": int(float(parts[1].strip())),
                    "driver_version": parts[2].strip()
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"nvidia-smi not found or failed: {e}")
    return None


def detect_cuda_version():
    """Detect installed CUDA toolkit version."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.4" from "Cuda compilation tools, release 12.4, V12.4.131"
                    import re
                    match = re.search(r"release\s+([\d.]+)", line)
                    if match:
                        return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def get_optimal_gpu_layers(vram_mb):
    """
    Determines optimal n_gpu_layers based on available VRAM.
    -1 means offload ALL layers to GPU (best performance).
    """
    if vram_mb >= 8192:
        return -1  # 8GB+ → full offload, plenty of room
    elif vram_mb >= 6144:
        return -1  # 6GB → full offload for 3B models (tight but works)
    elif vram_mb >= 4096:
        return 20  # 4GB → partial offload
    elif vram_mb >= 2048:
        return 10  # 2GB → minimal GPU assist
    else:
        return 0   # <2GB → CPU only


def get_gpu_config():
    """
    Returns the GPU configuration for the current system.
    
    Returns:
        dict with keys:
            - use_gpu (bool): Whether to use GPU acceleration
            - n_gpu_layers (int): Number of layers to offload (-1 = all)
            - gpu_info (dict|None): GPU hardware details
            - cuda_version (str|None): CUDA toolkit version
    """
    # Force CPU mode
    if USE_GPU == "false":
        logger.info("GPU disabled by USE_GPU=false in .env")
        return {
            "use_gpu": False,
            "n_gpu_layers": 0,
            "gpu_info": None,
            "cuda_version": None
        }

    gpu_info = detect_nvidia_gpu()
    cuda_version = detect_cuda_version()

    # Force GPU mode (will fail at runtime if no GPU, but user asked for it)
    if USE_GPU == "true":
        n_layers = -1
        if gpu_info:
            n_layers = get_optimal_gpu_layers(gpu_info["vram_mb"])
        return {
            "use_gpu": True,
            "n_gpu_layers": n_layers,
            "gpu_info": gpu_info,
            "cuda_version": cuda_version
        }

    # Auto-detect mode
    if gpu_info:
        n_layers = get_optimal_gpu_layers(gpu_info["vram_mb"])
        if n_layers > 0 or n_layers == -1:
            return {
                "use_gpu": True,
                "n_gpu_layers": n_layers,
                "gpu_info": gpu_info,
                "cuda_version": cuda_version
            }

    # No GPU detected → CPU fallback
    return {
        "use_gpu": False,
        "n_gpu_layers": 0,
        "gpu_info": None,
        "cuda_version": cuda_version
    }


# Cache the config at module load time
GPU_CONFIG = get_gpu_config()


def print_gpu_status():
    """Prints a human-readable GPU status report."""
    config = GPU_CONFIG
    print("\n" + "=" * 60)
    print("  GPU CONFIGURATION STATUS")
    print("=" * 60)

    if config["use_gpu"] and config["gpu_info"]:
        gpu = config["gpu_info"]
        print(f"  [+] NVIDIA GPU Detected: {gpu['name']}")
        print(f"     VRAM: {gpu['vram_mb']} MB | Driver: {gpu['driver_version']}")
        if config["cuda_version"]:
            print(f"     CUDA Toolkit: {config['cuda_version']}")
        else:
            print(f"     [!] CUDA Toolkit: Not found (install for best results)")
        
        layers = config["n_gpu_layers"]
        if layers == -1:
            print(f"     Mode: FULL GPU OFFLOAD (-1 layers = all)")
        else:
            print(f"     Mode: PARTIAL GPU OFFLOAD ({layers} layers)")
        print(f"\n  [OK] Inference will be GPU-accelerated!")
    elif config["use_gpu"]:
        print(f"  [~] GPU mode forced but no NVIDIA GPU detected")
        print(f"     Will attempt GPU offload (may fail at runtime)")
    else:
        print(f"  [-] Running in CPU-only mode")
        print(f"     Inference will be slower. To enable GPU:")
        print(f"     1. Install NVIDIA CUDA Toolkit 12.x")
        print(f"     2. Run: setup_gpu.bat")
        print(f"     3. Set USE_GPU=true in .env (optional)")
    
    print("=" * 60 + "\n")


# Print status on import during startup
if __name__ != "__main__":
    # Only log a concise message when imported as a module  
    if GPU_CONFIG["use_gpu"] and GPU_CONFIG["gpu_info"]:
        gpu_name = GPU_CONFIG["gpu_info"]["name"]
        logger.info(f"GPU Acceleration: ON ({gpu_name}, {GPU_CONFIG['n_gpu_layers']} layers)")
    else:
        logger.info("GPU Acceleration: OFF (CPU mode)")
