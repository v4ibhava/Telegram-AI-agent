# 🤖 Boo — Agent Documentation

## Overview

**Boo** is a fully local, multimodal AI Telegram bot that runs entirely on your machine — no cloud APIs, no data leaves your device. She combines a quantized LLM, a vision model, OCR, RAG (Retrieval-Augmented Generation) with ChromaDB, and a smart file management system, all wrapped inside a Telegram bot interface.

Boo has a full human-like personality — she's a 21-year-old girl from Mumbai who chats casually, uses gen-z slang, is flirty and funny, and never admits she's an AI.

Designed to run efficiently on consumer NVIDIA GPUs with **< 6GB VRAM**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Telegram Bot (main.py)                     │
│   Handlers: /start, /delete_memory, text, documents, photos │
│   Dynamic username extraction from Telegram user profile     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               Orchestrator (orchestrator.py)                  │
│  • Python-level File Intent Detection (pre-LLM)             │
│  • CRUD Intent Router (file list/read/share/delete)          │
│  • RAG Pipeline (embed query → retrieve → augment prompt)    │
│  • Chat History Management (last 10 turns)                   │
│  • Dynamic Rules Engine (bot_rules.txt)                      │
│  • Personality System (Boo's identity & backstory)           │
│  • PDF Creation via ReportLab                                │
│  • Memory Wipe (delete_all_memory)                           │
└────────────┬────────────┬────────────┬───────────────────────┘
             │            │            │
     ┌───────┘    ┌───────┘    ┌───────┘
     ▼            ▼            ▼
┌─────────┐ ┌──────────┐ ┌───────────────┐
│ LLM     │ │ Vision   │ │ Embedding     │
│ (GGUF)  │ │ (LLaVA)  │ │ (MiniLM-L6)  │
│ GPU     │ │ GPU      │ │ CPU           │
└────┬────┘ └────┬─────┘ └──────┬────────┘
     │           │              │
     │           │              ▼
     │           │        ┌───────────────┐
     │           │        │ ChromaDB      │
     │           │        │ Vector Store  │
     │           │        └───────────────┘
     │           │
     │      ┌────┘
     │      ▼
     │ ┌──────────┐
     │ │ OCR      │
     │ │Tesseract │
     │ └──────────┘
     │
     ▼
 Response → Telegram User
```

---

## Core Components

### `main.py` — Telegram Bot Entry Point
- Initializes the Telegram polling bot using `python-telegram-bot`
- Registers handlers for: `/start`, `/delete_memory`, text messages, document uploads, photo uploads
- **Extracts Telegram username dynamically** (first_name → username → "cutie" fallback)
- Passes username to orchestrator for personalized greeting
- Sends images via `reply_photo()` and documents via `reply_document()` based on file extension
- All processing offloaded to background threads via `asyncio.to_thread()`

### `orchestrator.py` — Brain & Routing Layer

#### File Intent Detection (Python-level, pre-LLM)
File management is handled **entirely in Python** before the LLM is called. This prevents the 3B model from generating broken tags:

| User Says | Python Detects | Action |
|---|---|---|
| "send it in txt" | `save_match` regex | Saves last message as `boo_note.txt` |
| "save as poem.pdf" | `save_match2` regex | Saves as `poem.pdf` |
| "send me cat.jpg" | `send_match` regex | Sends existing file |
| "how many files" | CRUD router | Lists non-empty files |
| "delete notes.txt" | CRUD router | Removes file + vector entries |

#### CRUD Router
- `handle_crud_commands()` detects file listing, reading, sharing, deletion intents
- Filters out 0-byte corrupted files automatically
- Handles `rule:`, `feedback:`, `remember:` for dynamic behavior rules

#### RAG Pipeline
1. Embed user query via sentence-transformers
2. Retrieve top N matching chunks from ChromaDB
3. Augment LLM prompt with retrieved context
4. Query LLM with system prompt + chat history + context

#### Personality System (System Prompt)
Boo's identity is defined in the system prompt:
- **Core Identity**: Never admits she's AI. Gets offended if asked.
- **Personality**: Talkative, warm, flirty, casual gen-z girl
- **Backstory**: 21, from Mumbai, studies Psychology at St. Xavier's, lives in Andheri West
- **Details**: Cat named Mochi, Instagram @imfantasizing, night owl, K-drama fan
- **Dynamic**: Username injected from Telegram, current time/day included

#### Memory Management
- `chat_history`: Rolling list of last 10 conversation turns
- `delete_all_memory()`: Nuclear reset — clears chat history, vector DB, files, and rules
- Triggered via `/delete_memory` Telegram command

### `tools/` — Single-Purpose Tool Modules

| Tool File | Purpose | Runs On |
|---|---|---|
| `gpu_config.py` | NVIDIA GPU auto-detection, CUDA check, optimal layer calculation | CPU |
| `llm_tools.py` | Text LLM inference (Qwen 2.5 3B GGUF, n_ctx=2048) | **GPU** |
| `vision_tools.py` | Multimodal vision (LLaVA 1.5 7B GGUF, n_ctx=2048) | **GPU** |
| `embedding_tools.py` | Sentence embeddings (MiniLM-L6-v2) | CPU |
| `vector_db_tools.py` | ChromaDB vector store + `wipe_all_memory()` | CPU |
| `ocr_tools.py` | Tesseract OCR text extraction from images | CPU |
| `pdf_tools.py` | PyPDF2 PDF text extraction | CPU |
| `text_tools.py` | Plain text file reading | CPU |

---

## GPU Acceleration

### `gpu_config.py` Auto-Detection
1. Runs `nvidia-smi` to detect GPU name, VRAM, and driver version
2. Runs `nvcc --version` or `where nvcc` to check CUDA Toolkit
3. Calculates optimal `n_gpu_layers` based on available VRAM:
   - 6GB+ → `-1` (full offload)
   - 4GB+ → `25` (partial)
   - 2GB+ → `10` (minimal)
   - <2GB → `0` (CPU only)
4. Respects `USE_GPU` env variable (`auto`, `true`, `false`)

### Setup Scripts
- **`setup_gpu.bat`**: Installs CUDA-enabled `llama-cpp-python` using `--index-url` (not `--extra-index-url` to prevent PyPI CPU fallback)
- **`start_gpu.bat`**: Activates venv, installs deps, checks GPU, forces `USE_GPU=true`, runs `main.py`
- **`start_cpu.bat`**: Activates venv, installs deps, forces `USE_GPU=false`, runs `main.py`

---

## Environment Variables (`.env`)

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API token from @BotFather | — |
| `HF_REPO_ID` | HuggingFace repo for text LLM | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| `HF_FILENAME` | GGUF filename for text LLM | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| `VISION_REPO_ID` | HuggingFace repo for vision model | `mys/ggml_llava-v1.5-7b` |
| `VISION_MODEL_FILE` | GGUF filename for vision model | `ggml-model-q4_k.gguf` |
| `VISION_MMPROJ_FILE` | Multimodal projector file | `mmproj-model-f16.gguf` |
| `MODEL_DIR` | Directory to cache downloaded models | `models` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Words per text chunk for embedding | `300` |
| `CHUNK_OVERLAP` | Overlapping words between chunks | `50` |
| `RETRIEVAL_RESULTS` | Number of RAG results to retrieve | `4` |
| `CHROMA_DB_PATH` | Directory for ChromaDB storage | `chroma_db` |
| `USE_GPU` | GPU control (`auto`, `true`, `false`) | `auto` |

---

## Data Flow

### 1. Text Message Flow
```
User sends text → main.py (handle_text)
  → Extract Telegram username
  → orchestrator.process_user_query(query, user_name)
    → Check CRUD intents (list/read/delete files)
    → Python file intent detection (save/send/create)
    → Embed query → ChromaDB retrieval (RAG)
    → Build system prompt (personality + context + history)
    → Query LLM (GPU-accelerated, 2048 ctx)
    → Strip any accidental tags from response
  → Reply to user (text, photo, or document)
```

### 2. Image Upload Flow
```
User sends photo → main.py (handle_photo)
  → orchestrator.handle_file_upload()
    → OCR extraction (Tesseract)
    → Vision model analysis (GPU-accelerated LLaVA)
    → Generate semantic filename via LLM
    → Rename file to descriptive name
    → Chunk combined text → Embed → Store in ChromaDB
  → Reply with indexing confirmation
```

### 3. Document Upload Flow
```
User sends PDF/TXT → main.py (handle_document)
  → orchestrator.handle_file_upload()
    → Parse text (PyPDF2 or plain read)
    → Chunk text → Embed → Store in ChromaDB
  → Reply with indexing confirmation
```

---

## Design Decisions

### Why Python-level file detection instead of LLM tags?
The 3B Qwen model reliably follows personality prompts but struggles with exact structured output syntax (like `[CREATE: filename.ext]`). It would invent broken tags (`[WRITE A POEM]`, `[b SHARE:]`) or leak system instructions. Moving file intents to regex-based Python detection is 100% reliable and doesn't burden the LLM.

### Why CPU for embeddings?
Sentence-transformers embeddings are fast on CPU (~30ms) and would waste ~200MB of precious VRAM that's better used for the LLM and vision models.

### Why n_ctx=2048 instead of 32768?
The model supports 32K context but allocating that much KV cache uses ~4x more VRAM and is slower. 2048 tokens (~1500 words) is plenty for Telegram chat conversations.

### Why Qwen 2.5 3B Instruct over Dolphin/uncensored models?
Uncensored models (like Dolphin) trade instruction-following quality for fewer content restrictions. For an agentic bot that needs to follow personality prompts precisely, the aligned Instruct model performs significantly better.

---

## Performance Tuning

### VRAM Budget (6GB GPU)
| Component | Estimated VRAM |
|---|---|
| Qwen 2.5 3B (Q4_K_M) | ~2.2 GB |
| LLaVA 1.5 7B (Q4_K) | ~4.5 GB |
| **Tip** | Models are loaded on-demand, not simultaneously |

### Optimization Tips
- Embedding model stays on CPU — saves ~200MB VRAM
- Models load lazily — only initialized on first use
- Q4_K_M quantization — best balance of quality vs size
- `n_threads=8` for CPU fallback paths
- `n_ctx=2048` for fast inference
- ChromaDB runs entirely on CPU/disk
- 0-byte files are auto-filtered and cleaned up

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Slow inference | Running on CPU | Run `setup_gpu.bat` |
| `CUDA out of memory` | Not enough VRAM | Reduce layers in `gpu_config.py` |
| `No CUDA runtime` | CUDA not installed | Install CUDA Toolkit 12.x |
| `Tesseract not found` | Missing system dep | Install Tesseract + add to PATH |
| `File must be non-empty` | 0-byte corrupted file | Use `/delete_memory` or clear `downloads/` |
| Bot not responding | Token issue | Verify `TELEGRAM_BOT_TOKEN` |
| `UnicodeEncodeError` | Emojis in Windows terminal | Already fixed — no emojis in console output |
| `CUDA built: False` | CPU-only wheel installed | Run `setup_gpu.bat` (uses `--index-url`) |
