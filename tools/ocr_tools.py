import pytesseract
from PIL import Image

def perform_ocr(file_path):
    """Extracts text from an image using Tesseract OCR."""
    try:
        img = Image.open(file_path)
        # Note: Tesseract must be installed on the system map
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"
