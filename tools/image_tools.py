from PIL import Image

def load_image(file_path):
    """Loads an image and returns basic properties."""
    try:
        img = Image.open(file_path)
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "path": file_path
        }
    except Exception as e:
        return {"error": str(e)}
