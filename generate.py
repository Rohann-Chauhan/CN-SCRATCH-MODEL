import torch

from model import GPTModel
from tokenizer import (
    encode,
    decode,
    vocab_size
)

from config import (
    device,
    max_seq_len
)

from utils import causal_mask

# GLOBAL INITIALIZATION (Prevents reloading model weights on every prompt)
print("Loading model weights, please wait...")
model = GPTModel(vocab_size).to(device)
model.load_state_dict(
    torch.load(
        "best_model.pt",
        map_location=device
    )
)
model.eval()


def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40
):
    # Convert your input text prompt into token ID numbers
    ids = encode(prompt)

    # Wrap the IDs inside a PyTorch tensor and send them to the GPU/CPU
    idx = torch.tensor(
        [ids],
        dtype=torch.long
    ).to(device)

    generated_text = ""

    # Disable gradient tracking since we are just generating text, not training
    with torch.no_grad():

        for _ in range(max_new_tokens):

            # If the text becomes longer than what the model can handle, trim it
            if idx.size(1) > max_seq_len:
                idx_cond = idx[:, -max_seq_len:]
            else:
                idx_cond = idx

            # Create a look-ahead mask for the current context size
            mask = causal_mask(idx_cond.size(1)).to(device)

            # Pass tokens into the model to get raw scoring numbers (logits)
            logits = model(idx_cond, mask)

            # Isolate the scoring values for only the very last predicted token
            logits = logits[:, -1, :]

            # Adjust creative freedom by dividing scores by temperature
            logits = logits / temperature

            # Isolate only the top K highest scoring options
            values, indices = torch.topk(logits, top_k)

            # Turn raw scores into readable percentage probabilities (0% to 100%)
            probs = torch.softmax(values, dim=-1)

            # Roll a weighted die to select the next token out of the top K options
            next_token = indices.gather(
                -1,
                torch.multinomial(probs, num_samples=1)
            )

            # Append the newly generated token ID to our running sequence
            idx = torch.cat([idx, next_token], dim=1)

            # Turn the token ID back into human-readable text
            token_text = decode(next_token[0].tolist())
            generated_text += token_text

            # Stop the loop early if the model writes its stopping phrase
            if "Question:" in generated_text:
                break

    # Final decode of the entire generated sequence
    result = decode(idx[0].tolist())

    # Strip away trailing data if the stopping phrase was triggered
    if "Question:" in result:
        result = result.split("Question:")[0]

    return result


if __name__ == "__main__":

    print("=" * 50)
    print("Custom GPT Inference Engine")
    print("=" * 50)

    # Run an interactive conversation loop in your terminal
    while True:
        prompt = input("\nPrompt: ")

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not prompt.strip():
            continue

        output = generate(
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40
        )

        print("\nGenerated:\n")
        print(output)