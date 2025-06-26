import sys
import os
import subprocess

def run_script(arg):
    if arg == "start":
        print("Starting the application...")
        # Replace with your actual start command
        subprocess.run(["python3", os.path.join(os.getcwd(),"runtime","app.py")])

    if arg == "train":
        print("Training the model...")
        # Replace with your actual training command
        subprocess.run(["python3", os.path.join(os.getcwd(),"train","train_data.py")])

if __name__ == "__main__":
    print(sys.argv)