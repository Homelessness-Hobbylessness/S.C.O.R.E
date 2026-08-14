import time
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    
    print("Swift Correction & Results Engine")
    
    time.sleep(3)
    
    clear_screen()
    image_path = input("Please paste the file path of the image: ")
    
    sys.exit()

if __name__ == "__main__":
    main()
