import os
import cv2
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
from torchvision import transforms, datasets
from data import get_metadata, get_dataset, fix_legacy_dict
import unets
from PIL import Image
from torchmetrics.functional.image import image_gradients

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
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        # MODIFICATION
        self, model, xT, style, timesteps=None, partial_steps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """

        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps

        if partial_steps:
            new_timesteps = np.linspace(
                0, partial_steps - 1, num=timesteps, endpoint=True, dtype=int
            )    
        else:
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
                
                # MODIFICATION
                pred_epsilon = model(final, current_t, style[0].repeat(len(final), 1, 1, 1), **model_kwargs)
                
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
    local_rank,
):
    model.train()
    graytransform = transforms.Grayscale()

    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(args.device) - 1,
            labels.to(args.device) if args.class_cond else None,
        )
        t = torch.randint(args.partial_steps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        
        # MODIFICATION
        style_indices = torch.randperm(images.shape[0])
        style_images = images[style_indices]
        pred_eps = model(xt, t, style_images, y=labels)
        
        # lambda_cyc: constant for the loss of the style image with the model output
        if args.loss_func == 'l1':
            # loss = ((pred_eps - eps) ** 2 + args.lambda_cyc * torch.abs(pred_eps - (xt - style_images))).mean()

            loss = ((pred_eps - eps) ** 2 + args.lambda_cyc * torch.abs(pred_eps - (images - style_images))).mean()

        elif args.loss_func == 'grad':
            input_grad = image_gradients(graytransform(images))
            style_grad = image_gradients(graytransform(style_images))

            input_mag = torch.sqrt(input_grad[0] ** 2 + input_grad[1] ** 2)
            style_mag = torch.sqrt(style_grad[0] ** 2 + style_grad[1] ** 2)

            loss = ((pred_eps - eps) ** 2 + args.lambda_cyc * torch.abs(pred_eps - (input_mag - style_mag))).mean()
        
        else:
            loss = ((pred_eps - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)


def sample_N_images(
    N,
    model,
    diffusion,
    # MODIFICATION
    style,
    xT=None,
    sampling_steps=250,
    
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    
    # print('\n', args, '\n')
    # print("\nSAMPLING\n")

    with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(args.device)
                )
            if args.class_cond:
                y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                    args.device
                )
            else:
                y = None
            gen_images = diffusion.sample_from_reverse_process(
                model=model, xT=xT, style=style, timesteps=sampling_steps, model_kwargs={"y": y}, ddim=args.ddim, partial_steps=args.partial_steps
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None, style[0])


def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
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
        help="Number of partial noising steps",
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
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
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
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )
        #MODIFICATION
    parser.add_argument("--lambda_cyc", type=float, default=0.05)
    parser.add_argument("--loss_func", type=str, default='l1')
    parser.add_argument("--style_image_path", type=str, default='')
    parser.add_argument("--input_images_path", type=str, default='')
    
    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--seed", default=112233, type=int)

    parser.add_argument("--outname", type=str, default=None)

    # setup
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("No Local Rank found, defaulting to 0.")
        local_rank = 0
    metadata = get_metadata(args.dataset)
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    if local_rank == 0:
        print(args)

    # Creat model and diffusion process
    model = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(args.device)
    if local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
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
    elif ngpus == 1:
        torch.distributed.init_process_group(backend='nccl', init_method="env://", world_size=1, rank=0)
        model = DDP(model, device_ids=[0], output_device=0)
        args.batch_size = args.batch_size // ngpus

    # sampling
    if args.sampling_only:
        style_transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor()
            ]
        )
        if args.style_image_path != '':
            try:
                style_image = Image.open(args.style_image_path)
                style_image = style_transform(style_image)
            except:
                print(f"Ooops something went wrong with the style image path you provided. \nPath provided: {args.style_image_path}")

        else:
            print(f"No style image provided. Please provide an image to generate samples.")
            return
        
        if args.input_images_path != '':
            if os.path.isdir(args.input_images_path):
                input_images = []
                for img_file in os.listdir(args.input_images_path):
                    input_images.append(style_transform(Image.open(args.input_images_path + img_file)).to(args.device))

                input_images = torch.stack(input_images, dim=0)

            else:
                input_images = style_transform(Image.open(args.input_images_path)).to(args.device)

            input_images, _ = diffusion.sample_from_forward_process(input_images, args.partial_steps)
            style_image = torch.stack([style_image] * len(input_images))

        else:
            print(f"No input image provided. Please provide an image to generate samples.")
            return

        sampled_images, labels, _ = sample_N_images(
            len(input_images),
            model,
            diffusion,
            style_image,
            input_images,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args,
        )

        if args.outname:
            output_name = args.outname
        else:
            output_name = f"{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps_lambda_cyc-{args.lambda_cyc}_epochs-{args.epochs}_partial-steps-{args.partial_steps}_loss-{args.loss_func}_images-class_condn_{args.class_cond}"

        np.savez(
            os.path.join(
                args.save_dir,
                f"{output_name}.npz",
            ),
            sampled_images,
            labels,
        )

        # print(sampled_images)

        cv2.imwrite(
            os.path.join(
                args.save_dir,
                f"{output_name}.png",
            ),
            np.concatenate(sampled_images, axis=1)[:, :, ::-1],
        )

        cv2.imwrite(
            os.path.join(
                args.save_dir,
                f"{output_name}_style_image.png",
            ),
            style_image[0].numpy().T * 255
        )
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    if local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
        )
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    args.ema_dict = copy.deepcopy(model.state_dict())

    # lets start training the model
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args, local_rank)
        if not epoch % 1:
            if args.style_image_path:
                style_transform = transforms.Compose(
                    [
                        transforms.Resize(64),
                        transforms.ToTensor()
                    ]
                )
                style_image = style_transform(Image.open(args.style_image_path))
            else:
                style_image = train_loader.dataset[np.random.randint(len(train_loader.dataset))]
            sampled_images, _, style_image = sample_N_images(
                64,
                model,
                diffusion,
                # MODIFICATION
                style_image,
                None,
                args.sampling_steps,
                args.batch_size,
                metadata.num_channels,
                metadata.image_size,
                metadata.num_classes,
                args,
            )
            if local_rank == 0:
                cv2.imwrite(
                    os.path.join(
                        args.save_dir,
                        f"{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps_lambda_cyc-{args.lambda_cyc}_epochs-{args.epochs}_partial-steps-{args.partial_steps}_loss-{args.loss_func}.png",
                    ),
                    np.concatenate(sampled_images, axis=1)[:, :, ::-1],
                )

                cv2.imwrite(
                    os.path.join(
                        args.save_dir,
                        f"{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps_lambda_cyc-{args.lambda_cyc}_epochs-{args.epochs}_partial-steps-{args.partial_steps}_loss-{args.loss_func}_style_image.png",
                    ),
                    style_image.numpy().T * 255
                )

        if local_rank == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    f"{args.dataset}-epoch_{args.epochs}-{args.sampling_steps}-sampling_steps_lambda_cyc-{args.lambda_cyc}_partial-steps-{args.partial_steps}_loss-{args.loss_func}.pt",
                ),
            )
            torch.save(
                args.ema_dict,
                os.path.join(
                    args.save_dir,
                    f"{args.dataset}-epoch_{args.epochs}-{args.sampling_steps}-sampling_steps_ema_{args.ema_w}_lambda_cyc-{args.lambda_cyc}_partial-steps-{args.partial_steps}_loss-{args.loss_func}.pt",
                ),
            )


if __name__ == "__main__":
    main()
