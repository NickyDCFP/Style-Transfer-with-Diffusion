import torch
from data_music import encode_spectrogram

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

        # must use [-1, 1] pixel range for music
        waves = [
            encode_spectrogram(tracks[i], metadata) for i in range(len(tracks))
        ]
        input_tracks = waves[:len(waves)//2]
        style_tracks = waves[len(waves)//2:]
        input_tracks = torch.stack(input_tracks).to(args.device)
        style_tracks = torch.stack(style_tracks).to(args.device)
        assert (input_tracks.max().item() <= 1)  and \
               (-1 <= input_tracks.min().item()) and \
               (style_tracks.max().item() <= 1)  and \
               (-1 <= style_tracks.min().item()),    \
            f"range is [{tracks.min().item(), tracks.max().item()}]"
        t_steps = args.partial_steps if args.use_style else args.diffusion_steps
        t = torch.randint(t_steps, (len(input_tracks),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(input_tracks, t)
        if args.use_style:
            pred_eps = model(xt, t, style_tracks)
            loss = ((pred_eps - eps) ** 2 + args.lambda_cyc * torch.abs(pred_eps - (xt - style_tracks))).mean()
        else:
            pred_eps = model(xt, t)
            loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()
        input_tracks = input_tracks.cpu()
        style_tracks = style_tracks.cpu()

        # update ema_dict
        if local_rank == 0:
            new_dict = model.state_dict()
            for (k, _) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)
