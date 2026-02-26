# 🤖 Boo — Local AI Telegram Agent (Under 6GB VRAM)

A fully local, private AI companion running as a Telegram bot. No cloud APIs, no data leaves your device. Powered by quantized LLMs with **NVIDIA CUDA GPU acceleration**, RAG memory, OCR, and vision — all on a consumer GPU.

**Boo** isn't just a chatbot — she's a talkative, friendly, gen-z girl who chats naturally, manages your files, remembers your conversations, and never admits she's an AI 😄

## ✨ Features
- **GPU Accelerated**: Auto-detects NVIDIA GPUs and offloads all LLM layers for blazing fast inference
- **Human-Like Personality**: Boo chats casually like a real person — flirty, funny, and emotionally expressive
- **Conversational Memory**: Remembers your name, chat history (last 10 turns), and stored context
- **Multimodal AI**: Upload images and Boo will analyze them locally via LLaVA vision model
- **RAG & Long-Term Memory**: Store text, PDFs, CSVs, and images into ChromaDB vector memory for context-aware answers
- **Smart File Management**: Save, share, delete files — all handled by Python-level intent detection (no broken LLM tags)
- **Self-Improving**: Send `rule:`, `feedback:`, or `remember:` to permanently adjust Boo's behavior
- **Memory Wipe**: Use `/delete_memory` to factory-reset all chat history, files, and vector memory

## 📂 Project Structure
```
├── main.py              # Telegram bot entry point (async handlers)
├── orchestrator.py      # Brain: CRUD router, RAG pipeline, file management, personality
├── agent.md             # Full architecture documentation
├── tools/
│   ├── gpu_config.py    # NVIDIA GPU auto-detection & CUDA configuration
│   ├── llm_tools.py     # Text LLM inference (Qwen 2.5 3B, GPU accelerated)
│   ├── vision_tools.py  # Vision model (LLaVA 1.5 7B, GPU accelerated)
│   ├── embedding_tools.py # Sentence embeddings (MiniLM-L6, CPU)
│   ├── vector_db_tools.py # ChromaDB persistent vector store + wipe_all_memory()
│   ├── ocr_tools.py     # Tesseract OCR extraction
│   ├── pdf_tools.py     # PyPDF2 PDF reader
│   └── text_tools.py    # Plain text file reader
├── .env                 # Configuration (bot token, model paths, GPU settings)
├── start_gpu.bat        # One-click GPU mode launcher
├── start_cpu.bat        # One-click CPU mode launcher
├── setup_gpu.bat        # NVIDIA CUDA setup script
└── requirements.txt     # Python dependencies
```

## 🔧 Prerequisites
1. **Python 3.10+**
2. **NVIDIA GPU** with 6GB+ VRAM (RTX 3050/3060 or better) — *optional, CPU mode available*
3. **NVIDIA Drivers** installed ([download](https://www.nvidia.com/drivers))
4. **CUDA Toolkit 12.x** ([download](https://developer.nvidia.com/cuda-downloads))
5. Telegram Bot Token from `@BotFather`
6. **Tesseract OCR** installed and added to System PATH

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/v4ibhava/Telegram-AI-agent.git
cd Telegram-AI-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure
Open `.env` and set your bot token:
```
TELEGRAM_BOT_TOKEN=your_token_here
```

### 3. GPU Setup (Recommended)
```bash
setup_gpu.bat
```
This auto-detects your GPU, installs the CUDA-enabled `llama-cpp-python`, and verifies GPU access.

### 4. Run
```bash
start_gpu.bat   # For NVIDIA GPUs
# OR
start_cpu.bat   # For systems without NVIDIA hardware
```

On startup you'll see:
```
============================================================
  GPU CONFIGURATION STATUS
============================================================
  [+] NVIDIA GPU Detected: NVIDIA GeForce RTX 3050 6GB Laptop GPU
     VRAM: 6144 MB | Driver: 591.74
     CUDA Toolkit: 12.1
     Mode: FULL GPU OFFLOAD (-1 layers = all)

  [OK] Inference will be GPU-accelerated!
============================================================
```

## 💬 Commands

| Command | Description |
|---|---|
| `/start` | Welcome message with capabilities |
| `/delete_memory` | Factory reset — wipes all chat history, files, vector memory, and rules |
| `rule: <instruction>` | Permanently save a behavior rule |
| `feedback: <text>` | Adjust Boo's behavior |
| `remember: <info>` | Store a persistent fact |

## 📁 File Management
File operations are handled automatically by Python intent detection — no need for special syntax:

| What you say | What happens |
|---|---|
| "how many files do you have" | Lists all non-empty files |
| "send it in txt" | Saves last message as `boo_note.txt` and sends it |
| "save as poem.pdf" | Saves last message as `poem.pdf` |
| "send me cat.jpg" | Sends the existing file |
| "delete notes.txt" | Removes from disk + vector memory |

## ⚡ Performance

### VRAM Budget (6GB GPU)
| Component | Estimated VRAM |
|---|---|
| Qwen 2.5 3B (Q4_K_M) | ~2.2 GB |
| LLaVA 1.5 7B (Q4_K) | ~4.5 GB |
| **Note** | Models load on-demand, not simultaneously |

### Optimization
- Context window set to 2048 tokens for fast inference
- Embedding model runs on CPU to save VRAM
- Models load lazily (only on first use)
- ChromaDB runs entirely on CPU/disk

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| Slow inference | Run `setup_gpu.bat` to install CUDA build |
| `CUDA out of memory` | Reduce layers in `gpu_config.py` or use smaller model |
| `No CUDA runtime` | Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) |
| `Tesseract not found` | Install Tesseract and add to PATH |
| Bot not responding | Check `TELEGRAM_BOT_TOKEN` in `.env` |
| `File must be non-empty` | Corrupted 0-byte files — use `/delete_memory` or delete `downloads/` |

## 🤝 Contributing
Contributions welcome! Fork, branch, and open a PR. Make sure new tools play nicely with `orchestrator.py`.

---
*Built with ❤️ — runs entirely on your machine, your data stays yours.*
