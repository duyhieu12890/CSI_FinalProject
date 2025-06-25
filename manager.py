import sys
import os
import subprocess

def run_script(arg):
    if arg == "start":
        print("Starting the application...")
        # Replace with your actual start command
        subprocess.run(["python3", os.path.join(os.getcwd(),"runtime","app.py")])
        
if __name__ == "__main__":
    print(sys.argv)