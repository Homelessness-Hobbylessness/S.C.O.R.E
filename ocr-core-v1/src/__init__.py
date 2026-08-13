import urllib.request
import sys

# The RAW URL to your actual OCR core script
OCR_SCRIPT_URL = "https://raw.githubusercontent.com/Homelessness-Hobbylessness/S.C.O.R.E/main/ocr-core-v1/src/__init__.py"

def start_score():
    print("=" * 55)
    print("  S.C.O.R.E. Launcher ")
    print("=" * 55)
    print("[+] Connecting to GitHub repository...")
    
    try:
        # Fetch the file using Python's built-in library (no dependencies needed)
        response = urllib.request.urlopen(OCR_SCRIPT_URL)
        script_code = response.read().decode('utf-8')
        
        print("[✓] OCR Core fetched successfully. Starting ...\n")
        print("-" * 55)
        
        # Execute the fetched Python code dynamically
        exec(script_code, globals())
        
    except urllib.error.HTTPError as e:
        print(f"\n[!] Download Failed (HTTP {e.code}).")
        print("    Make sure the repository is public or update the URL.")
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    start_score()
