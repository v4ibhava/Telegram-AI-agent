import PyPDF2

def parse_pdf(file_path):
    """Parses text from a PDF file."""
    text_content = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        return text_content
    except Exception as e:
        return f"Error parsing PDF: {e}"
