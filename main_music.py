import os
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio

from data_music import fix_legacy_dict, get_metadata, decode_spectrogram, encode_spectrogram, get_gtzan, collate_gtzan, decode_spectrogram_diffwave
import unets

unsqueeze3x = lambda x: x[..., None, None, None]


class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the music.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling music by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better music quality.

        Return: An music tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final


class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
    metadata,
    local_rank,
):
    model.train()
    for step, (tracks) in enumerate(dataloader):
        # assert (tracks.max().item() <= 1) and (-1 <= tracks.min().item()), f"range is [{tracks.min().item(), tracks.max().item()}]"

        # must use [-1, 1] pixel range for music
        tracks = torch.stack([
            encode_spectrogram(tracks[i], metadata) for i in range(len(tracks))
        ]).to(args.device)
        t = torch.randint(diffusion.timesteps, (len(tracks),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(tracks, t)
        pred_eps = model(xt, t)

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()
        tracks = tracks.cpu()

        # update ema_dict
        if local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)


def sample_N_tracks(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    height_dim=256,
    time_dim=2048,
    args=None,
):
    """use this function to sample any number of tracks from a given
        diffusion model and diffusion process.

    Args:
        N : Number of tracks
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the music.
        length_samples : music spectrogram in  time frames
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N tracks.
    """
    samples, num_samples = [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, height_dim, time_dim)
                    .float()
                    .to(args.device)
                )
            gen_music = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y" : None}, args.ddim
            )
            samples_list = [torch.zeros_like(gen_music) for _ in range(num_processes)]

            dist.all_gather(samples_list, gen_music, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = np.concatenate(samples)[:N]
    samples = torch.Tensor(samples)
    samples[samples == 0] = 1e-5
    # samples = np.concatenate(samples)[:N]
    # samples = (127.5 * (samples + 1)).astype(np.uint8)
    return samples

def sample_and_save_tracks(
    args,
    model,
    diffusion,
    metadata,

):
    sampled_specs = sample_N_tracks(
        args.num_sampled_tracks,
        model,
        diffusion,
        None,
        args.sampling_steps,
        args.batch_size,
        metadata.num_channels,
        height_dim=metadata.rescaled_height_dim,
        time_dim=metadata.time_dim,
        args=args,
    )
    if args.save_specs:
        torch.save(
            sampled_specs.squeeze(0),
            os.path.join(
                args.save_dir,
                f"spec_{args.sampling_steps}-sampling_steps.pt"
            )
        )
    # if args.no_diffwave:
    sampled_waves = torch.cat(
        [
            decode_spectrogram(sampled_specs[i, :, :, :], metadata)
            for i in range(sampled_specs.size(0))
        ],
        dim=1
    )
    sampled_diffwaves = torch.cat(
        [
            decode_spectrogram_diffwave(sampled_specs[i, :, :, :], metadata, args)
            for i in range(sampled_specs.size(0))
        ],
        dim=1
    )
    torchaudio.save(
        os.path.join(
            args.save_dir,
            f"sampling_steps-{args.sampling_steps}.wav",
        ),
        sampled_waves,
        metadata.sampling_rate
    )
    torchaudio.save(
        os.path.join(
            args.save_dir,
            f"sampling_steps-{args.sampling_steps}_diffwave.wav",
        ),
        sampled_diffwaves,
        metadata.sampling_rate
    )


def main():
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
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
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
        "--no-diffwave",
        action="store_true",
        default=False,
        help="Don't use diffwave to decode",
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
    metadata = get_metadata('gtzan')
    music = get_gtzan(args.data_dir)
    model = unets.__dict__[args.arch](
        time_dim=metadata.time_dim,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
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
        sample_and_save_tracks(args, model, diffusion, metadata)
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

    # lets start training the model
    for epoch in range(args.epochs):
        print("Epoch", epoch + 1)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args, metadata, local_rank)
        if (not epoch % 50) or (epoch == args.epochs - 1):
            sample_and_save_tracks(args, model, diffusion, metadata)
            if local_rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"epoch_{args.epochs}-{args.sampling_steps}-sampling_steps.pt",
                    ),
                )
                torch.save(
                    args.ema_dict,
                    os.path.join(
                        args.save_dir,
                        f"epoch_{args.epochs}-{args.sampling_steps}-sampling_steps_ema_{args.ema_w}.pt",
                    ),
                )

if __name__ == "__main__":
    main()
