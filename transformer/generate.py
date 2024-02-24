import argparse

import torch
import tiktoken

from src.generator import TextGenerator
from src.dataset import NaiveEncoder

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default=None, help="path to past model")
    parser.add_argument("--encode_scheme", type=str, default="naive", help="encoding scheme")
    parser.add_argument("--text_filepath", type=str, default="input.txt", help="path to text file")
    parser.add_argument("--output_filepath", type=str, default="output.txt", help="path to output file")

    # model
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--block_size", type=int, default=64, help="block size")
    parser.add_argument("--n_embd", type=int, default=256, help="embedding dimension")
    parser.add_argument("--n_head", type=int, default=4, help="number of heads")
    parser.add_argument("--n_inner", type=int, default=1024, help="inner dimension")
    parser.add_argument("--n_layer", type=int, default=3, help="number of layers")
    parser.add_argument("--feed_forward_dropout", type=float, default=0.5, help="feed forward dropout")
    parser.add_argument("--attn_dropout", type=float, default=0.5, help="attention dropout")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.model_checkpoint is not None, "model_checkpoint must be provided"

    if args.encode_scheme == "naive":
        with open(args.text_filepath, "r") as f:
            text = f.read()
        encoder = NaiveEncoder(text)
    else:
        encoder = tiktoken.get_encoding("cl100k_base")
    
    args_dict = vars(args)
    args_dict["n_vocab"] = encoder.n_vocab

    generator = TextGenerator(
        model_checkpoint=args.model_checkpoint,
        block_size=args.block_size,
        decoder_args=args_dict,
    )

    context = torch.zeros(1, 1, dtype=torch.long)
    tokens = generator.generate(context)[0].tolist()


    sample = encoder.decode(tokens)

    print(sample)

    with open(args.output_filepath, "w") as f:
        f.write(sample)
    
    