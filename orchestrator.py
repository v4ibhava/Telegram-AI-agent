import os
from dotenv import load_dotenv
from tools.text_tools import parse_text
from tools.pdf_tools import parse_pdf
from tools.ocr_tools import perform_ocr
from tools.vision_tools import analyze_image
from tools.embedding_tools import chunk_text, get_embedding
from tools.vector_db_tools import store_in_memory, retrieve_from_memory, delete_by_source
from tools.llm_tools import query_llm

load_dotenv()
RETRIEVAL_RESULTS = int(os.getenv("RETRIEVAL_RESULTS", 4))

def handle_file_upload(file_path, file_name, file_type):
    """
    Orchestrates the ingestion, processing, OCR/Vision extraction, 
    chunking, embedding, and memory storage of uploaded context.
    CRUD: CREATE operation with automatic index integration.
    """
    extracted_text = ""
    metadata = {"source": file_name, "type": file_type}
    
    print(f"Processing '{file_name}' of type '{file_type}'...")
    
    try:
        if file_type == 'text/plain':
            extracted_text = parse_text(file_path)
            
        elif file_type == 'application/pdf':
            extracted_text = parse_pdf(file_path)
            
        elif file_type.startswith('image/'):
            # Multimodal approach: extract OCR text and a descriptive Vision caption
            ocr_text = perform_ocr(file_path)
            vision_caption = analyze_image(file_path, prompt="Describe the image, extracting meaningful details and transcribing any visible large text.")
            
            extracted_text = f"Image Name: {file_name}\n"
            extracted_text += f"---\nOCR Transcription:\n{ocr_text}\n"
            extracted_text += f"---\nAI Vision Description:\n{vision_caption}\n"
        
        else:
            return f"Unsupported file type: {file_type} for {file_name}."
        
        # If successfully extracted context, create chunks and store in Vector DB (ChromaDB)
        if extracted_text and extracted_text.strip():
            chunks = chunk_text(extracted_text)
            stored_count = 0
            
            for chunk in chunks:
                emb = get_embedding(chunk)
                doc_id = store_in_memory(chunk, emb, metadata)
                if doc_id:
                    stored_count += 1
            
            return f"Successfully processed '{file_name}'. Indexed {stored_count} chunks into long-term memory."
        else:
            return f"Could not extract meaningful content from '{file_name}'."
            
    except Exception as e:
        return f"Error during orchestrator file handling: {e}"


def handle_crud_commands(user_query):
    """Parses lightweight CRUD intents from user for file management."""
    uq_lower = user_query.lower()
    import re
    
    # Check for listing files
    if "how many files" in uq_lower or "list files" in uq_lower:
        if not os.path.exists("downloads"):
            return "I currently have 0 files.", None
            
        files = os.listdir("downloads")
        if not files:
            return "I currently have 0 files in my folder.", None
            
        file_list = "\n".join([f"{i+1}. `{f}`" for i, f in enumerate(files)])
        return f"I have {len(files)} files in my directory:\n{file_list}", None

    # Check for reading files
    read_intent = re.search(r'(?:whats inside|read|what is in).*?([\w-]+\.\w+)', uq_lower)
    if read_intent:
        file_name = read_intent.group(1)
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path):
            # Only read text files
            if file_name.endswith('.txt') or file_name.endswith('.md') or file_name.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return f"Contents of `{file_name}`:\n\n{content}", None
            else:
                return f"`{file_name}` is not a text file. If you want it, ask me to 'send' it instead.", None
        return f"I couldn't find `{file_name}` in my folder.", None
        
    # Check for resharing files
    import re
    reshare_intent = re.search(r'(?:reshare|share|send me|send).*?([\w-]+\.\w+)', uq_lower)
    if reshare_intent:
        file_name = reshare_intent.group(1)
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path):
            return f"Here is the file you requested: {file_name}", file_path
        return f"I couldn't find `{file_name}` in my folder to send.", None

    # Check for file deletions
    delete_intent = re.search(r'delete.*?([\w-]+\.\w+)', uq_lower)
    if delete_intent:
        file_name = delete_intent.group(1)
        
        # Safely delete from the folder
        file_path = os.path.join("downloads", file_name)
        file_deleted = False
        if os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
            
        # Delete from Vector Memory
        db_deleted = delete_by_source(file_name)
        
        if file_deleted or db_deleted:
            return f"Successfully deleted `{file_name}` from disk and AI memory.", None
        else:
            return f"Could not find `{file_name}` to delete.", None

    if uq_lower.startswith("delete "):
        file_name = user_query[7:].strip()
        delete_by_source(file_name)
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        return f"Deleted all embeddings and files for '{file_name}'.", None
        
    return None


# Initialize a simple conversation history memory
chat_history = []

def process_user_query(user_query):
    """
    RAG orchestrated flow:
    1. Checks for direct system commands (CRUD Delete)
    2. Converts query to embedding 
    3. Retrieves top N matching context chunks
    4. Instructs Local LLM using augmented context + chat history array
    """
    global chat_history
    
    # 1. Direct Intent checking (CRUD tool router)
    crud_response = handle_crud_commands(user_query)
    if crud_response:
        return crud_response
        
    # 2. RAG Retrieval phase
    query_emb = get_embedding(user_query)
    retrieved_context = retrieve_from_memory(query_emb, n_results=RETRIEVAL_RESULTS)
    context_str = "\n\n".join(retrieved_context) if retrieved_context else ""
    
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
    
    # Read actual files in directory to give AI true context of available files
    current_files = []
    if os.path.exists("downloads"):
        current_files = os.listdir("downloads")
    file_list_str = "\n".join([f"- {f}" for f in current_files]) if current_files else "No files currently uploaded."
    
    # 3. LLM Generation
    system_msg = {
        "role": "system",
        "content": (
            "You are an intelligent multimodal local assistant on Telegram named 'Local AI Agent'.\n"
            f"The current local date and time is: {current_time}.\n"
            "The user talking to you is named 'abey'. Always address him naturally.\n\n"
            "You have full permissions over the 'downloads/' folder on your host machine to create, send, delete, or search files. "
            "If the user asks you to create a file (e.g., 'write a poem in poem.txt'), reply ONLY with the content meant for that file.\n"
            "If the user asks how to share/get an image or file, tell them to literally type 'share [filename]' to trigger the backend system correctly (e.g. 'share photo.jpg'). Don't offer them bash commands like Tesseract.\n"
            f"\n--- Currently Available Files in your 'downloads/' Folder ---\n{file_list_str}\n"
            "\nIf the user asks questions about past files, refer to the provided 'Context retrieved from memory'."
        )
    }
    
    messages = [system_msg]
    
    # Append the last 10 messages from history to retain maximum natural context
    for msg in chat_history[-10:]:
        messages.append(msg)
    
    # Bundle the latest user request with vector database context if anything was retrieved
    if context_str:
        final_prompt = (
            f"Context retrieved from memory:\n{context_str}\n\n"
            f"User Question:\n{user_query}"
        )
    else:
        final_prompt = user_query
        
    # Append the newest user intent
    messages.append({"role": "user", "content": final_prompt})
    
    response = query_llm(messages)
    
    # Save the bare query (no context strings) and response into short-term memory
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": response})
    
    # 4. Check if the user wanted a file to be returned
    # Crude implementation: if user says "write in poem.txt", try to parse a filename
    import re
    file_intent = re.search(r'(?:send|write|save).*? ([\w-]+\.txt)', user_query.lower())
    if file_intent:
        file_name = file_intent.group(1)
        os.makedirs('downloads', exist_ok=True)
        out_path = os.path.join('downloads', file_name)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(response)
        return f"I generated the content. Sending it over!", out_path
    
    return response, None
