import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from orchestrator import handle_file_upload, process_user_query

# Load .env variables
load_dotenv()

# Bot Token (You will need to replace this with your actual bot token, e.g. from BotFather)
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Silence the noisy auto-polling HTTP logs so your terminal stays clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the bot interaction."""
    welcome_text = (
        "Hello! I am your Multimodal Local AI Assistant running entirely on < 6GB VRAM.\n\n"
        "Capabilities:\n"
        "- Chat with me across any topics.\n"
        "- Upload text, PDF, and Images for CRUD storage and long-term memory.\n"
        "- My Vision Module will 'see' your images alongside an OCR pass.\n"
        "- My RAG system will index files to answer questions about any uploads.\n\n"
        "Go ahead, talk to me or send a file!"
    )
    await update.message.reply_text(welcome_text)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends normal chatter to the RAG orchestrator."""
    user_query = update.message.text
    logger.info(f"Received query: {user_query}")
    
    await update.message.chat.send_action(action="typing")
    response, out_file_path = await asyncio.to_thread(process_user_query, user_query)
    
    if out_file_path and os.path.exists(out_file_path):
        await update.message.reply_document(document=open(out_file_path, 'rb'), caption=response)
    else:
        await update.message.reply_text(response)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives and processes PDFs and Text files."""
    doc = update.message.document
    file = await context.bot.get_file(doc.file_id)
    
    # Store locally for the pipeline to parse
    os.makedirs('downloads', exist_ok=True)
    file_path = os.path.join('downloads', doc.file_name)
    await file.download_to_drive(file_path)
    
    msg = await update.message.reply_text(f"Received {doc.file_name}. Extracting text and embedding to memory...")
    
    # Orchestrator handles processing & chunking & storage
    result = await asyncio.to_thread(handle_file_upload, file_path, doc.file_name, doc.mime_type)
    
    await msg.edit_text(result)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives photos, triggers Vision, OCR, and embedding pipelines."""
    # Telegram sends multiple photo sizes, we take the largest one
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    
    os.makedirs('downloads', exist_ok=True)
    file_name = f"photo_{photo.file_id}.jpg"
    file_path = os.path.join('downloads', file_name)
    await file.download_to_drive(file_path)
    
    msg = await update.message.reply_text("Received image. Looking at contents (OCR + Vision model) and committing to vector memory...")
    
    result = await asyncio.to_thread(handle_file_upload, file_path, file_name, "image/jpeg")
    
    await msg.edit_text(result)

if __name__ == '__main__':
    print("--- Local AI Telegram Agent initializing ---")
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or not BOT_TOKEN:
        print("WARNING: Please set TELEGRAM_BOT_TOKEN in the .env file.")
        
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("\n" + "="*60)
    print(" 🚀 LOCAL AI TELEGRAM AGENT IS SUCCESSFULLY LOADED! 🚀 ")
    print("         Waiting for your messages on Telegram...        ")
    print("="*60 + "\n")
    application.run_polling()
