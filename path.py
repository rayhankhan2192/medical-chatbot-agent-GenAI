import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_paths = [
    "scripts/__init__.py",
    "scripts/cli.py",   
    "scripts/main.py",
    "scripts/agent.py",
    "setup.py",
]

for path_str in list_of_paths:
    path = Path(path_str)
    if not path.exists():
        logging.info(f"Creating path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()  
    else:
        logging.info(f"Path already exists: {path}")