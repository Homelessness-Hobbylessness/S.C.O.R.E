import os
import sys
from PIL import Image
import pytesseract

def get_image_path():
    while True:
        path = input("Please paste the file path of the image: ").strip().strip('"')
        if os.path.isfile(path):
            try:
                Image.open(path).verify()
                return path
            except Exception:
                pass
        print("Invalid image file. Try again.")

def main():
    path = get_image_path()
    text = pytesseract.image_to_string(Image.open(path))
    print(text.strip() or "(No text detected)")
    sys.exit()

if __name__ == "__main__":
    main()
