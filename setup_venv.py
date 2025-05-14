import subprocess
import sys
from pathlib import Path

# workspace base folder
venv_path = Path(__file__).resolve().parent.parent.parent / ".venv"
repo_base_path = Path(__file__).resolve().parent

if not venv_path.exists():
    print(f"Creating a new virtual environment at {venv_path}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

pip = venv_path / "bin" / "pip"

print("Installing dependencies..")

subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True)
subprocess.run([str(pip), "install", "-r", f"{repo_base_path}/requirements.txt"], check=True)

print("Virtual environment automatically successfully created and initialized!")