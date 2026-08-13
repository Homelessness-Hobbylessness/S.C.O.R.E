import urllib.request
import os
import sys


OCR_SCRIPT_URL = "https://raw.githubusercontent.com/Homelessness-Hobbylessness/S.C.O.R.E/main/ocr-core-v1/src/__init__.py"

def clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

def start_ocr_engine():
    clear_screen()
    print("=" * 55)
    print(" OCR v1")
    print("=" * 55)
    print("[+] Connecting to GH")
    
    try:
        response = urllib.request.urlopen(OCR_SCRIPT_URL)
        script_code = response.read().decode('utf-8')
        
        print("[✓] Loaded successfully\n")
        print("-" * 55)
        
       
        exec(script_code, {})
        
    except urllib.error.HTTPError as e:
        print(f"\n[!] Download fehlgeschlagen (HTTP {e.code}).")
    except Exception as e:
        print(f"\n[!] An error occured: {e}")
        

    input("\n[ Press enter to return to main screen ]")

def main_menu():
    while True:
        clear_screen()
        print("=" * 55)
        print("       S.C.O.R.E.       ")
        print("=" * 55)
        print(" [1] S.C.O.R.E. OCR ")
        print(" [2] Info")
        print(" [3] Beenden")
        print("-" * 55)
        
        choice = input(" choose (1-3): ").strip()
        
        if choice == '1':
            start_ocr_engine()
        elif choice == '2':
            clear_screen()
            print("=" * 55)
            print(" S.C.O.R.E.")
            print(" Made for schools")
            print(" optized for local data laws in germany")
            print("=" * 55)
            input("\n[ press enter to return ]")
        elif choice == '3':
            clear_screen()
            print("Stopping process \n")
            sys.exit(0)
        else:
            input("\n[!] Unknown input")

if __name__ == "__main__":

    main_menu()
