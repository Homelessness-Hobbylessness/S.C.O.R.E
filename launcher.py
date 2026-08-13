import urllib.request
import os
import sys

OCR_SCRIPT_URL = "https://raw.githubusercontent.com/Homelessness-Hobbylessness/S.C.O.R.E/main/ocr-core-v1/src/__init__.py"

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_ocr():
    clear()
    print("fetching ...")
    try:
        response = urllib.request.urlopen(OCR_SCRIPT_URL)
        code = response.read().decode('utf-8')
        print("running execution:\n" + "-" * 40)
        exec(code, {})
    except Exception as e:
        print(f"\n! error: {e}")
    input("\npress enter to return")

def main():
    while True:
        clear()
        print("s.c.o.r.e. // launcher\n")
        print("  1) start ocr")
        print("  2) about")
        print("  3) exit\n")
        
        choice = input("select > ").strip()
        
        if choice == '1':
            run_ocr()
        elif choice == '2':
            clear()
            print("nothing here rn")
            print("empty")
            input("press enter to return")
        elif choice == '3':
            clear()
            sys.exit(0)

if __name__ == "__main__":
    main()
