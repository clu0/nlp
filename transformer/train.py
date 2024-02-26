import argparse
from src.trainer import DecoderTrainer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--block_size", type=int, default=64, help="block size")
    parser.add_argument("--val_frac", type=float, default=0.1, help="validation fraction")
    parser.add_argument("--encode_scheme", type=str, default="naive", help="encoding scheme")
    parser.add_argument("--text_filepath", type=str, default="datasets/input.txt", help="path to text file")

    # training
    parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--save_interval", type=int, default=10000, help="save model interval")
    parser.add_argument("--lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--n_iter", type=int, default=500000, help="number of batch iterations")
    parser.add_argument("--n_val_iter", type=int, default=100, help="number of validation batches")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="path to past model")
    parser.add_argument("--past_n_iter",type=int,default=None,help="n iterations for past model checkpoint",)
    parser.add_argument("--model_save_dir", type=str, default="models/", help="prefix for saving model")
    parser.add_argument("--log_save_dir", type=str, default="logs/", help="prefix for saving model")
    parser.add_argument("--gradient_clip", type=float, default=None, help="gradient clipping value")

    # model
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
    
    trainer = DecoderTrainer(
        args=vars(args),
    )

    trainer.train(
        n_iter=args.n_iter,
        lr=args.lr,
        n_val_iter=args.n_val_iter,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        model_checkpoint=args.model_checkpoint,
        past_n_iter=args.past_n_iter,
        gradient_clip=args.gradient_clip,
    )