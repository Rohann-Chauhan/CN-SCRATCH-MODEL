from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token


def tokenize(text):
    return tokenizer.tokenize(text)


def encode(text):
    return tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False
    )


def decode(ids):
    return tokenizer.decode(ids)

