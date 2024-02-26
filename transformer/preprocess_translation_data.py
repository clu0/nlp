"""
Preprocesses the WMT 14 german to english translation data, downloaded from here: https://nlp.stanford.edu/projects/nmt/

will use the cl100k_base encoding scheme from tiktoken

a good fraction of the training dataset will have below 50 tokens after encoding,
so we will choose only the text that encodes to 50 tokens or less, and pad everything to have length 50

we will pad with the token 198, which represents the next line character
"""
import tiktoken
from time import time
import os
import torch
import numpy as np


BATCH_SIZE = 50
PAD_TOKEN = 198
DATA_DIR = "datasets/wmt-14-en-de"
MAX_BLOCK = 1000000  # max number of lines to process at once

if __name__ == "__main__":
    en_filename = os.path.join(DATA_DIR, "train.en")
    de_filename = os.path.join(DATA_DIR, "train.de")

    encoder = tiktoken.get_encoding("cl100k_base")
    
    cur_ind = 0
    valid_inds = []
    en_tokens = []
    de_tokens = []
    start_time = time()
    with open(en_filename, "r") as f_en , open(de_filename, "r") as f_de:
        for en_line, de_line in zip(f_en, f_de):
            en_encoding = encoder.encode(en_line)
            de_encoding = encoder.encode(de_line)
            if len(en_encoding) <= BATCH_SIZE and len(de_encoding) <= BATCH_SIZE:
                valid_inds.append(cur_ind)
                if len(en_encoding) < BATCH_SIZE:
                    en_encoding += [PAD_TOKEN] * (BATCH_SIZE - len(en_encoding))
                if len(de_encoding) < BATCH_SIZE:
                    de_encoding += [PAD_TOKEN] * (BATCH_SIZE - len(de_encoding))
                en_tokens.append(en_encoding)
                de_tokens.append(de_encoding)
            cur_ind += 1

            if cur_ind % 100000 == 0:
                print(f"processed {cur_ind} lines, took {time() - start_time:.2f} seconds")
            
            if cur_ind % MAX_BLOCK == 0:
                print(f"saving at {cur_ind} lines")
                en_tokens = torch.tensor(en_tokens, dtype=torch.long)
                de_tokens = torch.tensor(de_tokens, dtype=torch.long)
                ind_suffix = f"_{cur_ind - MAX_BLOCK}_{cur_ind}"
                torch.save(en_tokens, os.path.join(DATA_DIR, f"en_tokens{ind_suffix}.pt"))
                torch.save(de_tokens, os.path.join(DATA_DIR, f"de_tokens{ind_suffix}.pt"))
                selected_inds = np.array(valid_inds)
                np.save(os.path.join(DATA_DIR, f"selected_inds{ind_suffix}.npy"), selected_inds)
                valid_inds = []
                en_tokens = []
                de_tokens = []
            # for testing
            #if cur_ind > 100:
            #    break
            