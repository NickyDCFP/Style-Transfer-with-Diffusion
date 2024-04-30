# CATlation: Style Transfer with Diffusion Models!

Codebase for Advanced ML Project with Professor Anna Choromanska. Paper is available in [report.pdf]

Heavily based on a [previous paper](https://arxiv.org/abs/2105.03117) that used GANs for the same task.

To use, download the AFHQ dataset (not v2) from [here](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq). Put the `train` and `val` folders into `/dataset/misc/afhq256/`.


To train the model, run the following command:
```
torchrun --nproc_per_node=4 main.py --dataset afhq --arch UNet --epochs 300 --batch-size 20 --sampling-steps 50 --data-dir <DATA DIRECTORY> --lambda_cyc=0.005 --loss_func='l1' --save-dir <SAVE DIRECTORY> --partial-steps 250
```

To test the model:
```
torchrun --nproc_per_node=4 main.py --dataset afhq --arch UNet --batch-size 20 --style_image_path <PATH TO STYLE IMAGE> --input_images_path <PATH TO INPUT IMAGES> --pretrained-ckpt <OPTIONAL: checkpoint> --partial-steps 250  --outname <OUTPUT FILE NAME> --sampling-only 
```

Check the `music_diffusion` branch for style transfer in music.

