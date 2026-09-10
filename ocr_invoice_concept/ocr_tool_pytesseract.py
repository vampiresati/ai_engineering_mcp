from PIL import Image
import pytesseract
from langchain.tools import tool


@tool
def ocr_read_document_pytesseract(image_path: str) -> str:
    """Reads an image and returns text extracted using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error reading image: {e}"
if __name__=="__main__":
    ocr_output = ocr_read_document_pytesseract.invoke("receipt.jpg")
    print("=" * 80)
    print("RAW OCR OUTPUT")
    print("=" * 80)

    print(ocr_output)
