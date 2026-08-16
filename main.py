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

def main():
    path = get_image_path()
    text = pytesseract.image_to_string(Image.open(path))
    print(text.strip() or "(No text detected)")
    sys.exit()

if __name__ == "__main__":
    main()
