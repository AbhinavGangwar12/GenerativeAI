import json
from pathlib import Path

# Define the directory path as an object
info_path = Path("info")

# Check if the directory exists and is actually a directory
if info_path.is_dir():
    # Iterate through all files in the directory
    for file_path in info_path.iterdir():
        
        # Handle Markdown files
        if file_path.suffix == ".md":
            content = file_path.read_text(encoding="utf-8")
            print(f"Content of {file_path.name}:\n{content}\n")
            
        # Handle JSON files
        elif file_path.suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # indent=4 makes the printed JSON readable instead of a single long line
                print(f"Content of {file_path.name}:\n{json.dumps(data, indent=4)}\n")
