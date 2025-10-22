import os
import platform
import subprocess
from pathlib import Path

def run_unix_setup(project_dir):
    subprocess.run(["python", "docs/installation/setup_env.py", project_dir], check=True)

def run_windows_setup():
    subprocess.run(["docs\installation\install_env.bat"], shell=True)

if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent.parent)

    if platform.system() == "Windows":
        print("🪟 Detected Windows")
        run_windows_setup()
    else:
        print("🐧 Detected Unix/Linux/macOS")
        run_unix_setup(project_root)
