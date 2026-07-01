import pandas as pd
import json

def convert_to_jsonl():
    df = pd.read_csv("simple_cleaned_issues.csv")
    
    formatted_data = []
    for _, row in df.iterrows():
        # This is the "Instruction" structure the LLM will learn to follow
        entry = {
            "instruction": "Classify the following GitHub issue based on its title and body.",
            "input": f"Title: {row['title']}\nBody: {row['body']}",
            "output": row['simple_label']
        }
        formatted_data.append(entry)
    
    # Save as JSONL for training
    with open("training_data.jsonl", "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully converted {len(formatted_data)} records to training_data.jsonl")

if __name__ == "__main__":
    convert_to_jsonl()