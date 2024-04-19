import os
import copy
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data_music import fix_legacy_dict, get_metadata, get_gtzan, collate_gtzan
from train import train_one_epoch
from sample import sample_and_save_tracks
from diffusion import GaussianDiffusion
from logger import loss_logger
import unets
from args import parse_args

def main():
    args = parse_args()

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("No Local Rank found, defaulting to 0.")
        local_rank = 0
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    if local_rank == 0:
        print(args)
    metadata = get_metadata()
    music = get_gtzan(args.data_dir)
    model = unets.__dict__[args.arch](
        time_dim=metadata.time_dim,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        use_style=args.use_style
    ).to(args.device)
    diffusion = GaussianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model
    if args.pretrained_ckpt:
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
        dm = model.state_dict()
        if args.delete_keys:
            for k in args.delete_keys:
                print(
                    f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) doesn't match "
                    + f"with shape in model ({dm[k].shape})"
                )
                del d[k]
        model.load_state_dict(d, strict=False)
        print(
            f"Mismatched keys in ckpt and model: ",
            set(d.keys()) ^ set(dm.keys()),
        )
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.sampling_only:
        sample_and_save_tracks(args, model, diffusion, metadata, music)
        return

    # Load dataset
    sampler = DistributedSampler(music) if ngpus > 1 else None
    train_loader = DataLoader(
        music,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_gtzan
    )
    if local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Samples of music: {len(music)}"
        )
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    args.ema_dict = copy.deepcopy(model.state_dict())

    # start training the model
    for epoch in range(args.epochs):
        print("Epoch", epoch + 1)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args, metadata, local_rank)
        if (not epoch % 50) or (epoch == args.epochs - 1):
            sample_and_save_tracks(args, model, diffusion, metadata, music)
            if local_rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"lambda_{args.lambda_cyc}-{args.sampling_steps}-sampling_{args.partial_steps}-partial-use_style-{args.use_style}.pt",
                    ),
                )
                torch.save(
                    args.ema_dict,
                    os.path.join(
                        args.save_dir,
                        f"lambda_{args.lambda_cyc}-{args.sampling_steps}-sampling_{args.partial_steps}-partial-use_style-{args.use_style}_ema_{args.ema_w}.pt",
                    ),
                )

if __name__ == "__main__":
    main()
