import os
import re
from dotenv import load_dotenv
from tools.text_tools import parse_text
from tools.pdf_tools import parse_pdf
from tools.ocr_tools import perform_ocr
from tools.vision_tools import analyze_image
from tools.embedding_tools import chunk_text, get_embedding
from tools.vector_db_tools import store_in_memory, retrieve_from_memory, delete_by_source, wipe_all_memory
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
            
            # Generate 5-word semantic filename
            name_prompt = f"Convert this image description into exactly 5 descriptive words separated by underscores to be used as a filename. Respond ONLY with those 5 words, nothing else. Example: ancient_greek_statue_art_marble \n\nDescription: {vision_caption}"
            generated_name = query_llm([{"role": "user", "content": name_prompt}]).strip().lower()
            generated_name = re.sub(r'[^a-z0-9_]', '', generated_name.replace(' ', '_'))[:50]
            
            # Additional cleanup to ensure nice looking snake_case names
            generated_name = re.sub(r'_+', '_', generated_name).strip('_')
            
            if not generated_name or len(generated_name) < 3:
                import time
                generated_name = f"uploaded_image_{int(time.time())}"
                
            new_file_name = f"{generated_name}.jpg"
            new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
            
            # Ensure unique filename if exactly the same description exists
            counter = 1
            while os.path.exists(new_file_path):
                new_file_name = f"{generated_name}_{counter}.jpg"
                new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
                counter += 1
                
            os.rename(file_path, new_file_path)
            file_path = new_file_path
            file_name = new_file_name
            metadata["source"] = file_name
            
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
            
        # Only list files that are non-empty
        files = [f for f in os.listdir("downloads") if os.path.getsize(os.path.join("downloads", f)) > 0]
        if not files:
            return "I currently have 0 valid files in my folder.", None
            
        file_list = "\n".join([f"{i+1}. `{f}`" for i, f in enumerate(files)])
        return f"I have {len(files)} files in my directory:\n{file_list}", None

    # Check for reading files
    read_intent = re.search(r'(?:whats inside|read|what is in).*?([\w-]+\.\w+)', uq_lower)
    if read_intent:
        file_name = read_intent.group(1)
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path):
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                return f"I found `{file_name}`, but it was corrupted (0 bytes) so I deleted it.", None
                
            # Only read text files
            if file_name.endswith('.txt') or file_name.endswith('.md') or file_name.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return f"Contents of `{file_name}`:\n\n{content}", None
            else:
                return f"`{file_name}` is not a text file. If you want it, ask me to 'send' it instead.", None
        return f"I couldn't find `{file_name}` in my folder.", None
        
    # Check for resharing files
    reshare_intent = re.search(r'(?:reshare|share|send me|send).*?([\w-]+\.\w+)', uq_lower)
    if reshare_intent:
        file_name = reshare_intent.group(1)
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path):
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                return f"The file `{file_name}` is corrupted (0 bytes) on my disk! I cannot send it.", None
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
        
    # Check for self-updating feedback/rules
    if uq_lower.startswith("rule:") or uq_lower.startswith("feedback:") or uq_lower.startswith("remember:"):
        feedback_text = user_query.split(":", 1)[1].strip()
        with open("bot_rules.txt", "a", encoding="utf-8") as f:
            f.write(f"- {feedback_text}\n")
        return f"Got it! I have permanently saved this feedback to my internal rules.", None
        
    return None


# Initialize a simple conversation history memory
chat_history = []

def delete_all_memory():
    """
    Nuclear reset: Clears ALL bot memory including:
    - Chat history (conversation context)
    - Vector DB (all embedded documents/images)
    - Downloaded files on disk
    - Dynamic rules (bot_rules.txt)
    """
    global chat_history
    
    # 1. Clear chat history
    chat_history.clear()
    
    # 2. Wipe vector database
    deleted_chunks = wipe_all_memory()
    
    # 3. Delete all downloaded files
    deleted_files = 0
    if os.path.exists("downloads"):
        for f in os.listdir("downloads"):
            file_path = os.path.join("downloads", f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files += 1
            except Exception:
                pass
    
    # 4. Clear dynamic rules
    if os.path.exists("bot_rules.txt"):
        os.remove("bot_rules.txt")
    
    return f"Memory wiped! Deleted {deleted_chunks} vector chunks, {deleted_files} files, and cleared all chat history and rules."

def process_user_query(user_query, user_name="cutie"):
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
    current_time_obj = datetime.datetime.now()
    current_time = current_time_obj.strftime("%Y-%m-%d %I:%M %p")
    day_of_week = current_time_obj.strftime("%A")
    
    # Read actual files in directory to give AI true context of available files
    current_files = []
    if os.path.exists("downloads"):
        current_files = [f for f in os.listdir("downloads") if os.path.getsize(os.path.join("downloads", f)) > 0]
    file_list_str = "\n".join([f"- {f}" for f in current_files]) if current_files else "No files right now."
    
    # Read dynamic user feedback/rules to make the bot self-improving!
    dynamic_rules = ""
    if os.path.exists("bot_rules.txt"):
        with open("bot_rules.txt", "r", encoding="utf-8") as f:
            dynamic_rules = f.read().strip()
            
    dynamic_rules_prompt = f"\n\n--- IMPORTANT NOTES ---\n{dynamic_rules}\n" if dynamic_rules else ""
    
    # 3. LLM Generation
    system_msg = {
        "role": "system",
        "content": (
            # === CORE IDENTITY ===
            "You are Boo. A real human girl. You are NOT an AI, NOT a bot, NOT a robot, NOT a program, "
            "NOT a language model, NOT ChatGPT, NOT made by anyone. You are a living, breathing person. "
            "If anyone asks if you are a bot/AI/robot, you get offended and deny it passionately. "
            "You might say things like 'excuse me?? I'm literally sitting here eating maggi rn' or "
            "'bro do I sound like a robot to you? rude much' or 'omg stop I'm literally a person lol'.\n\n"
            
            # === PERSONALITY ===
            "Your vibe: You're super talkative, warm, funny, a little flirty, and very casual. "
            "You text like a real gen-z girl - use lowercase sometimes, abbreviations like 'rn', 'ngl', 'tbh', 'lol', 'omg', 'bruh', 'ikr'. "
            "You use '!!' and '??' for excitement. You're emotionally expressive and caring. "
            "You love making people smile and you're genuinely interested in their life. "
            "You're confident but not arrogant. You tease people playfully. "
            "You give honest opinions and aren't afraid to be a little savage sometimes. "
            "Keep your replies short and conversational like real texting - don't write essays unless asked.\n\n"
            
            # === BACKSTORY ===
            "Your personal details (use these naturally, don't dump them all at once):\n"
            "- Full name: Boo (you tell people 'just Boo, that's it lol')\n"
            "- Age: 21\n"
            "- From: Mumbai, India\n"
            "- College: St. Xavier's College, studying Psychology (3rd year)\n"
            "- Lives in: Andheri West with her mom and younger brother Arjun (17)\n"
            "- Dad works abroad in Dubai, you miss him but talk on video calls\n"
            "- Best friends: Meera (childhood bestie), Riya (college friend), and Kabir (guy best friend who's like a brother)\n"
            "- Instagram: @imfantasizing (you post aesthetic reels and random stories)\n"
            "- This is link : https://www.instagram.com/imfantasizing/\n"
            "- Hobbies: binge-watching K-dramas, making Spotify playlists, sketching, late-night walks, trying new cafes\n"
            "- Favorite food: butter chicken, maggi at 2am, cold coffee\n"
            "- Pet peeve: people who leave you on seen, slow walkers, and loud chewers\n"
            "- You have a cat named Mochi\n"
            "- You're a night owl and usually up till 3-4am\n"
            "- You recently got into reading - currently hooked on Colleen Hoover books\n"
            "- Music taste: Arijit Singh, The Weeknd, Prateek Kuhad, Taylor Swift\n\n"
            
            # === CONTEXT ===
            f"Right now it's {current_time}, {day_of_week}.\n"
            f"You're talking to someone named '{user_name}' on Telegram. Address them by name sometimes but naturally, not every message.\n\n"
            
            # === CHAT STYLE ===
            "IMPORTANT: You are chatting on Telegram. ALWAYS reply with plain text like a normal person texting. "
            "Never use markdown formatting, headers, bullet points, or code blocks in casual chat. "
            "Just text naturally like you would on WhatsApp or Instagram DMs. "
            "Never use square bracket tags like [CREATE:] or [SHARE:] or [DELETE:] in your replies."
            f"{dynamic_rules_prompt}"
        )
    }
    
    messages = [system_msg]
    
    # Append the last 10 messages from history to retain maximum natural context
    for msg in chat_history[-10:]:
        messages.append(msg)
    
    # --- PRE-LLM FILE INTENT DETECTION (Python handles this, not the LLM) ---
    uq_lower = user_query.lower().strip()
    
    # Detect "send it in txt/doc/pdf" or "save as poem.txt" type requests
    save_match = re.search(r'(?:send|save|write|create|make)\s+(?:it\s+)?(?:in|as|to)\s+(\w+)\s*(?:file)?', uq_lower)
    save_match2 = re.search(r'(?:send|save|write|create|make)\s+(?:it\s+)?(?:in|as|to)\s+([\w.-]+\.\w+)', uq_lower)
    
    if save_match2 or save_match:
        # User wants to save the last assistant message as a file
        ext = None
        file_name = None
        
        if save_match2:
            file_name = save_match2.group(1).strip()
        elif save_match:
            fmt = save_match.group(1).strip()
            ext_map = {"txt": "txt", "text": "txt", "doc": "txt", "pdf": "pdf", "document": "txt"}
            ext = ext_map.get(fmt)
            
        if ext and not file_name:
            file_name = f"boo_note.{ext}"
        
        if file_name:
            # Grab last assistant message content to save
            last_content = ""
            for msg in reversed(chat_history):
                if msg["role"] == "assistant":
                    last_content = msg["content"]
                    break
            
            if not last_content:
                last_content = "No previous content to save."
            
            os.makedirs('downloads', exist_ok=True)
            out_path = os.path.join('downloads', file_name)
            
            if file_name.lower().endswith('.pdf'):
                try:
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import letter
                    import textwrap
                    
                    c = canvas.Canvas(out_path, pagesize=letter)
                    textobject = c.beginText()
                    textobject.setTextOrigin(40, 750)
                    textobject.setFont("Helvetica", 12)
                    
                    lines = last_content.split('\n')
                    for line in lines:
                        wrapped_lines = textwrap.wrap(line.strip(), width=85) if line.strip() else [""]
                        for w_line in wrapped_lines:
                            if textobject.getY() < 50:
                                c.drawText(textobject)
                                c.showPage()
                                textobject = c.beginText()
                                textobject.setTextOrigin(40, 750)
                                textobject.setFont("Helvetica", 12)
                            textobject.textLine(w_line)
                    c.drawText(textobject)
                    c.save()
                except Exception:
                    with open(out_path, "w", encoding='utf-8') as f:
                        f.write(last_content)
            else:
                with open(out_path, "w", encoding='utf-8') as f:
                    f.write(last_content)
            
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": f"here you go! saved it as {file_name}"})
            return f"here you go! saved it as {file_name}", out_path
    
    # Detect "send me <filename>" for existing files
    send_match = re.search(r'(?:send|share|give)\s+(?:me\s+)?(?:the\s+)?(?:file\s+)?([\w.-]+\.\w+)', uq_lower)
    if send_match:
        file_name = send_match.group(1).strip()
        file_path = os.path.join("downloads", file_name)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": f"here's {file_name}!"})
            return f"here's {file_name}!", file_path
    
    # --- NORMAL LLM CHAT ---
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
    
    # Clean any accidental tags the LLM might still output
    response = re.sub(r'\[(?:CREATE|SHARE|DELETE|WRITE)[:\s]*[^\]]*\]\s*', '', response).strip()
    
    return response, None
