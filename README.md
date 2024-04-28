# Contrastive Learning for Unsupervised Music-to-Music Translation, with Diffusion Models!

Heavily based on a [previous paper](https://arxiv.org/abs/2105.03117) that used GANs for the same task.

Makes heavy use of code from [minimal-diffusion](https://github.com/VSehwag/minimal-diffusion) and lmnt-com's [diffwave](https://github.com/lmnt-com/diffwave/).

To use, download the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset. This setup will assume that GTZAN is installed in `./dataset/GTZAN/`.

First, set up diffwave.
```
cd diffwave
pip install .
```

Then, preprocess the GTZAN data and train Diffwave.
```
python -m diffwave.preprocess ./dataset/GTZAN/
python -m diffwave ./trained_models/ ./dataset/GTZAN/
```

At the same time, you can train the diffusion models themselves to generate music.
```
torchrun --nproc_per_node=4 main_music.py --arch UNet --batch-size 32 --sampling-steps 50 --partial-steps 250 --lambda-cyc 0.1
```
