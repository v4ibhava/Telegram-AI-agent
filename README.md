# Local AI Vision & RAG Agent (Under 6GB VRAM)

This project wraps a powerful Local-LLM into a private Telegram Bot. Designed completely around resource efficiency for consumer GPUs (like 6GB VRAM models) executing completely locally.
It integrates local Vector databases, OCR engines, local embedding models, and local quantized LLMs.
It supports advanced multimodal LLMs like `qwen2.5vl:3b`.

## 🧠 Features
- **Conversational Memory**: Remembers your name, the current time, and exactly what you were just talking about!
- **Multimodal AI**: Upload images and the AI will analyze them locally.
- **RAG & Memory Hook**: Ask the system to store Text, CSVs, PDFs, and images into its Long-Term memory so it acts like a true agentic vault. 
- **System Interactions**: Seamlessly ask the system to build files, search inside its directory, or even send locally stored scripts/images back to your phone directly from the chat interface!

## 📂 Project Structure
- `main.py`: Entrypoint for the Telegram Polling bot. Receives messages/documents and operates asynchronously.
- `orchestrator.py`: Orchestrator / reasoning layer. Manages prompts, chat history, and routes CRUD toolings.
- `tools/`: Extensible directory for all single-purpose tools.
  - `vector_db_tools.py`: Local ChromaDB persistent integration.
  - `embedding_tools.py`: Minimal sentence-transformers execution loaded into CPU.
  - `ocr_tools.py`: PyTesseract extraction tooling.
  - `pdf_tools.py`: PyPDF2 reading tooling for PDF document loads.
  - `text_tools.py`: Normal string extraction handling.
  - `vision_tools.py`: Interacts with locally ran VLM instances.
  - `llm_tools.py`: Interfaces with purely local LLMs via REST (Chat completions).
- `.env`: Central configuration file for bot token, model names, and DB configurations.
- `start.bat`: Convenience batch file to spin up python instances safely on Windows!

## 🔧 Prerequisites
1. Installed **Ollama** on your background OS natively.
   - Run `ollama pull qwen2.5vl:3b` (this acts as both the conversational LLM and the Vision Model setting).
2. Obtain a Telegram Bot Token locally using the `@BotFather` on Telegram.
3. Install **Tesseract** OCR engine natively and append it to System Path.
   
## 🚀 Installation
Ensure you are in the application directory:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set up your `.env`:
1. Open `.env`
2. Change `TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE` to your exact token.
3. Keep default settings for `qwen2.5vl:3b` mapped in configuring your LLM variables!

Run the bot natively:
```bash
start.bat
# OR
python main.py
```

## 🤝 Contributing
Contributions are absolutely welcome! We want to expand this lightweight AI to handle even cooler agentic local capabilities while remaining highly resource-efficient.

### How to contribute:
1. **Fork the repo** and clone it to your local machine.
2. **Create a branch** for your feature (`git checkout -b feature/AddCoolCapability`).
3. If you map a new tool into `tools/`, make sure it plays nicely with `orchestrator.py`.
4. **Commit your changes** (`git commit -m 'Added xyz function'`).
5. **Push to the branch** (`git push origin feature/AddCoolCapability`).
6. Open a **Pull Request**, explaining what your feature adds and its VRAM cost context!

*Feel free to report any bugs in the issues tab!*
