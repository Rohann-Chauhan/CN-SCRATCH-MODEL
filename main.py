import json
import os

from tokenizer import (
    encode,
    vocab_size
)

from training import (
    train_model
)


def main():
    # Define file path
    dataset_path = "train.json"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find '{dataset_path}'. Please make sure your JSON data "
            f"is saved with this exact filename in the current folder."
        )

    # Load your question-answer dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_tokens = []

    # Loop over your dataset items and format them into a structured text string
    for item in data:
        text = (
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n"
        )

        # Convert the raw text characters into token ID numbers
        ids = encode(text)
        
        # Add these IDs to our master training pool
        all_tokens.extend(ids)

    # Diagnostic statistics display
    print("=" * 50)
    print("Dataset Preprocessing Complete")
    print("=" * 50)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Total Tokens:    {len(all_tokens)}")
    print(f"First 20 Tokens: {all_tokens[:20]}")
    print("=" * 50)

    # Pass the token list directly into the training system
    train_model(
        all_tokens,
        vocab_size
    )


if __name__ == "__main__":
    main()