import json
import os
from training import train_model
from tokenizer import (
    encode,vocab_size
)
# Now open txt
all_tokens=[]
def main():
    datasset_path=PY_AJSON
    if not in os.path.exist(datasset_path):
        FileExistsError(f"the give file is not exise try another{datasset_path}")
    else:
        with open("datasset_path","r") as f:
            data=json.load(f)
    for item in data:
        text:(
            f"qustion":{item[question]}\n
            f"answer":{item[answer]}\n           
        )
    idx=encode(text)
    all_tokens.extend(idx)
    # Now call the model
    opeing=train_model(vocab_size,all_tokens)
    return opeing


if __name__ == "__main__":
    main()
