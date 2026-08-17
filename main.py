import os
import sys
import re
from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import Output

def get_image_path():
    while True:
        path = input("Paste file path: ").strip().strip('"')
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    img.load()
                return path
            except (UnidentifiedImageError, OSError):
                pass
        print("Invalid file")

def run_ocr(path):
    try:
        with Image.open(path) as img:
            data = pytesseract.image_to_data(img, timeout=30, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not installed.")
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(f"OCR failed: {e}")
        sys.exit(1)

    words, confidences = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = float(conf)
        if word and conf >= 0:
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    accuracy = sum(confidences) / len(confidences) if confidences else 0.0
    return text, accuracy

def text_ok(text, accuracy, min_accuracy=50.0):
    if not text.strip() or accuracy < min_accuracy:
        return False
    letters = sum(c.isalnum() for c in text)
    if letters / len(text) < 0.4:
        return False
    return bool(re.findall(r"[A-Za-z]{2,}", text))

def main():
    path = get_image_path()
    text, accuracy = run_ocr(path)

    print(f"Accuracy: {accuracy:.1f}%")

    if not text_ok(text, accuracy):
        print("No accurate text found.")
    else:
        print(text)

    sys.exit()

if __name__ == "__main__":
    main()
