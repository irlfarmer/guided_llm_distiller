import json
import tkinter as tk
from tkinter import filedialog
import re

def open_file():
    """Opens a file dialog to select a text or markdown file."""
    root = tk.Tk()
    root.withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt *.md")])
    return file_path

def parse_qa_file(file_path):
    """Parses a Q&A formatted file and converts it into OpenAI fine-tuning JSONL format."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into question-answer pairs using regex
    qa_pairs = re.findall(r"Question:\s*(.*?)\n(?:A|Answer):\s*(.*?)(?=\nQuestion:|\Z)", content, re.DOTALL)

    jsonl_data = []

    for question, answer in qa_pairs:
        entry = {
            "messages": [
                {"role": "system", "content": "You are an expert AI assistant providing clear and concise answers."},
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": answer.strip()}
            ]
        }
        jsonl_data.append(entry)

    return jsonl_data

def save_jsonl(data, output_path):
    """Saves the formatted JSONL data."""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    print("Select your Q&A data file...")
    file_path = open_file()

    if not file_path:
        print("No file selected. Exiting.")
    else:
        print(f"Processing file: {file_path}")

        jsonl_data = parse_qa_file(file_path)
        output_file = file_path.replace(".md", ".jsonl").replace(".txt", ".jsonl")

        save_jsonl(jsonl_data, output_file)
        print(f"JSONL file saved at: {output_file}")
