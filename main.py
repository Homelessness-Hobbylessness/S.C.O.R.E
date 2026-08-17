import os
import sys
from PIL import Image, UnidentifiedImageError
import pytesseract

def get_image_path():
    while True:
        path = input("Please paste the file path of the image: ").strip().strip('"')
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    img.load()
                return path
            except (UnidentifiedImageError, OSError):
                pass
        print("Invalid image file. Try again.")

def run_ocr(path):
    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img, timeout=30)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not on PATH.")
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(f"OCR failed: {e}")
        sys.exit(1)

def main():
    path = get_image_path()
    text = run_ocr(path)
    print(text.strip() or "(No text detected)")
    sys.exit()

if __name__ == "__main__":
    main()
