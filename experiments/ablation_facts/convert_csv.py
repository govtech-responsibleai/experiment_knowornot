import json
import pandas as pd
from pathlib import Path

def convert_json_to_csv(json_path: Path, input_dir: Path, output_dir: Path):
    # Load the JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract questions
    question_entries = data.get("questions", [])
    if not question_entries:
        print(f"[Warning] No questions found in {json_path}")
        return

    # Create a DataFrame
    df = pd.DataFrame(question_entries)[["identifier", "question", "answer"]]

    # Get relative path from input_dir and replace suffix with .csv
    relative_path = json_path.relative_to(input_dir).with_suffix('.csv')
    output_path = output_dir / relative_path

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    if not output_path.exists():
        df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[✓] Saved CSV to {output_path}")


if __name__ == "__main__":
    input_dir = Path("experiments/ablation_facts/questions")
    output_dir = Path("experiments/ablation_facts/questions_csv")
    # input_dir = Path("questions")
    # output_dir = Path("questions_csv")

    # Recursively find all JSON files
    for json_file in input_dir.rglob("*.json"):
        convert_json_to_csv(json_file, input_dir, output_dir)
