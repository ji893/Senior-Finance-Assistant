from PIL import Image
import pytesseract

def extract_text(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang='kor')
