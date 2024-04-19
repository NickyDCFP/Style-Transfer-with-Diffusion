import os
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import math
from data_music import decode_spectrogram, encode_spectrogram

def sample_N_tracks_unconditional(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=50,
    batch_size=32,
    num_channels=1,
    height_dim=128,
    time_dim=1024,
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

def sample_tracks_style(
    model,
    diffusion,
    args,
    input_tracks,
    style_tracks,
):
    samples, N, num_samples = [], input_tracks.size(0), 0
    batch_size = args.batch_size
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
        xT, _ = diffusion.sample_from_forward_process(
            input_tracks[num_samples:(num_samples + batch_size), :],
            args.partial_steps,
        )
        xT = xT.to(args.device)
        gen_music=diffusion.sample_from_reverse_process(
            model,
            xT,
            timesteps=args.sampling_steps,
            model_kwargs={"style" : style_tracks},
            partial=args.partial_steps
        )
        samples_list = [torch.zeros_like(gen_music) for _ in range(num_processes)]

        dist.all_gather(samples_list, gen_music, group)
        samples.append(torch.cat(samples_list).detach().cpu().numpy())
        num_samples += len(xT) * num_processes
        pbar.update(1)
    samples = np.concatenate(samples)
    samples = torch.Tensor(samples)
    samples[samples == 0] = 1e-5
    return samples

def sample_and_save_tracks(
    args,
    model,
    diffusion,
    metadata,
    dataset=None
):
    if args.use_style:
        inp_indices = np.random.randint(0, len(dataset), args.num_sampled_tracks)
        input_tracks = torch.stack([
            encode_spectrogram(dataset[ind][0], metadata) for ind in inp_indices
        ]).to(args.device)
        style_indices = np.random.randint(0, len(dataset), args.num_sampled_tracks)
        style_tracks = torch.stack([
            encode_spectrogram(dataset[ind][0], metadata) for ind in style_indices
        ]).to(args.device)
        sampled_specs = sample_tracks_style(
            model,
            diffusion,
            args,
            input_tracks,
            style_tracks
        )
        save_tracks(
            input_tracks,
            args,
            f"lambda_{args.lambda_cyc}-{args.sampling_steps}-sampling_{args.partial_steps}-partial-input",
            metadata,
            False
        )
        save_tracks(
            style_tracks,
            args,
            f"lambda_{args.lambda_cyc}-{args.sampling_steps}-sampling_{args.partial_steps}-partial-style",
            metadata,
            False
        )
    else:
        sampled_specs = sample_N_tracks_unconditional(
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
    save_tracks(
        sampled_specs,
        args,
        f"lambda_{args.lambda_cyc}-{args.sampling_steps}-sampling_{args.partial_steps}-partial-use-style-{args.use_style}",
        metadata,
        args.save_specs,
    )

def save_tracks(specs, args, filename, metadata, save_specs=False):
    sampled_waves = torch.cat(
        [
            decode_spectrogram(specs[i, :, :, :], metadata, args)
            for i in range(specs.size(0))
        ],
        dim=1
    )
    if save_specs:
        torch.save(
            specs.squeeze(0),
            os.path.join(
                args.save_dir,
                f"spec_{filename}.pt"
            )
        )
    torchaudio.save(
        os.path.join(
            args.save_dir,
            f"{filename}.wav",
        ),
        sampled_waves,
        metadata.sampling_rate
    )