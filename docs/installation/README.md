# 🛠 Tools

This folder contains utility scripts to set up the development environment for the project.

## Quick Start

Run the launcher to automatically detect your OS and set up a virtual environment:

```bash
python docs/installation/setup_env_launcher.py
```

### What it does

- Creates a `.venv/` virtual environment
- Installs all dependencies from `requirements.txt`
- Prints instructions on how to activate the environment

### Platform-specific scripts (used by the launcher)

- `setup_env.py` — for Linux/macOS
- `setup_env.bat` — for Windows
- `setup_env_launcher.py` — chooses the correct one based on your OS

## Manual Activation

### On Linux/macOS:
```bash
source .venv/bin/activate
```

### On Windows:
```cmd
call .venv\Scripts\activate.bat
```

Then run the app:
```bash
python main.py
```
