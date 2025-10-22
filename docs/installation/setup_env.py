import os
import subprocess
import sys
from pathlib import Path

def setup_virtualenv(project_dir):
    venv_path = Path(project_dir) / ".venv"
    python_exe = sys.executable

    if venv_path.exists():
        print("✅ Virtual environment already exists.")
    else:
        print("🚀 Creating virtual environment...")
        subprocess.run([python_exe, "-m", "venv", str(venv_path)], check=True)
        print(f"✅ Created virtual environment at: {venv_path}")

    pip_exe = venv_path / "bin" / "pip" if os.name != "nt" else venv_path / "Scripts" / "pip.exe"
    req_file = Path(project_dir) / "requirements.txt"

    if req_file.exists():
        print("📦 Installing requirements...")
        subprocess.run([str(pip_exe), "install", "-r", str(req_file)], check=True)
        print("✅ Dependencies installed.")
    else:
        print("⚠️ No requirements.txt found.")

    print("\n💡 Next steps:")
    if os.name == "nt":
        print(f"Run: {venv_path}\\Scripts\\activate.bat")
    else:
        print(f"Run: source {venv_path}/bin/activate")

    print("Then run your app with:")
    print("  python main.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Set up a virtual environment and install dependencies.")
    parser.add_argument("project_dir", help="Path to the root of the project (e.g. ./nano-xanes-viewer)")
    args = parser.parse_args()
    setup_virtualenv(args.project_dir)
