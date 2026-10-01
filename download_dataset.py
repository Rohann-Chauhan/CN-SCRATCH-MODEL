import os
from datasets import load_dataset


def download_and_save_dataset():
    output_filename = "train.txt"

    print("Downloading TinyStories dataset from Hugging Face...")
    # This downloads the dataset safely. It only downloads once; 
    # if you run it again, it will use a cached local copy.
    dataset = load_dataset("roneneldan/TinyStories")

    print(f"Extracting and saving data to '{output_filename}'...")
    
    # Open the file once and stream text straight to disk line-by-line
    with open(output_filename, "w", encoding="utf-8") as f:
        for row in dataset["train"]:
            f.write(row["text"] + "\n")

    print("=" * 50)
    print(f"Success! Dataset saved safely as: {output_filename}")
    print(f"File size: {os.path.path.getsize(output_filename) / (1024 * 1024):.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    download_and_save_dataset()