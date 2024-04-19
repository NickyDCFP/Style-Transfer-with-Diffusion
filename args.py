import argparse

def parse_args():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture")
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--partial-steps",
        type=int,
        default=250,
        help="Number of timesteps for partial noising of tracks"
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        default=False,
        help="Unconditional Generation"
    )
    # dataset
    parser.add_argument("--data-dir", type=str, default="./dataset/GTZAN/")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    parser.add_argument("--lambda-cyc", type=float, default=0.01)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample music (will save it in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-tracks",
        default=6,
        type=int,
        help="Number of music tracks to sample"
    )
    parser.add_argument(
        "--save-specs",
        action="store_true",
        default=False,
        help="Store generated spectrograms as well"
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()
    args.use_style = not args.unconditional
    return args