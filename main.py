import os
import sys
import time
import re
from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import Output

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_language():
    try:
        available = pytesseract.get_languages(config='')
 
except pytesseract.TesseractNotFoundError:
        print("Tesseract not installed.")
        sys.exit(1)

    while True:
        lang = input("OCR language (e.g. eng): ").strip()
        if lang in available:
            return lang
        print("Invalid language. Available languages:")
        print(", ".join(available))

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
        print("Invalid file. Try again.")

def run_ocr(path, lang):
    try:
        with Image.open(path) as img:
            data = pytesseract.image_to_data(img, lang=lang, timeout=30, output_type=Output.DICT)
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
    clear_screen()
    print("Swift Correction & Results Engine")
    time.sleep(2)
    clear_screen()

    lang = get_language()
    path = get_image_path()
    text, accuracy = run_ocr(path, lang)

    print(f"Accuracy: {accuracy:.1f}%")

    if not text_ok(text, accuracy):
        print("No real text found.")
    else:
        print(text)

    sys.exit()

if __name__ == "__main__":
    main()
