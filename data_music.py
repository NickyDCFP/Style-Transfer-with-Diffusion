import subprocess
import os
from easydict import EasyDict
from pathlib import Path
from datasets import load_dataset, Audio
from collections import OrderedDict
import torch
import torchaudio.transforms
from torchaudio.datasets import GTZAN
import torch.nn.functional as F
import librosa
from diffwave.inference import predict as diffwave_predict

def get_metadata(dataset: str) -> EasyDict:
    if dataset == "musiccaps":
        return EasyDict({
            "sampling_rate" : 44100,
            "length_secs" : 10,
            "train_waves" : 2663,
            "test_waves" : 2858,
            "num_channels" : 2,
            "time_dim" : 1024,
            "n_mels" : 80,
            "rescaled_height_dim" : 128,
            "vanilla_time_dim" : 862,
            "dynamic_range" : [-16, 8],
        })
    if dataset == "gtzan":
        return EasyDict({
            "sampling_rate" : 22050,
            "length_secs" : 20,
            "num_channels" : 1,
            "time_dim" : 1024,
            "n_mels" : 80,
            "rescaled_height_dim" : 128,
            "vanilla_time_dim" : 862,
            "dynamic_range" : [-16, 8],
        })

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False
    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" --force-keyframes-at-cuts "{url_base}{video_identifier}"
    """.strip()

    attempts = 0
    while True:
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def get_musiccaps(
    data_dir: str,
    metadata: EasyDict,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.

    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """
    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, _ = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path # probably reconfigure to none or something if the download failed if filter doesn't work
        example['download_status'] = status
        return example

    ds = ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).filter(lambda x: x['download_status'] == True).cast_column('audio', Audio(sampling_rate=metadata.sampling_rate, mono=False))
    ds = ds.with_format("torch")
    length_samples = metadata.sampling_rate * metadata.length_secs
    audio_stack = []
    for i in range(len(ds['audio'])):
        if ds['download_status'][i] != True:
            continue
        wave = ds['audio'][i]['array']
        if wave.size(0) != 2:
            wave = wave.repeat(2, 1)
        if wave.size(1) < length_samples:
            continue
        wave = wave[:, :length_samples]
        # spec = encode_spectrogram(wave, metadata)
        audio_stack.append(wave)
    return torch.stack(audio_stack)

def get_gtzan(data_dir: str):
    dataset = GTZAN(data_dir, download=False)
    return dataset

def collate_gtzan(minibatch):
    metadata = get_metadata('gtzan')
    length_samples = metadata.sampling_rate * metadata.length_secs
    return [
        sample[0][:, :length_samples]
        for sample in minibatch if sample[1] == metadata.sampling_rate
    ]

def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d

def encode_spectrogram(wave, metadata):
    transform = torchaudio.transforms.Spectrogram(
        n_fft=metadata.time_dim,
        power=2
    )
    t2 = torchaudio.transforms.MelScale(
        n_mels=metadata.n_mels,
        sample_rate=metadata.sampling_rate,
        n_stft=metadata.time_dim // 2 + 1
    )
    spec = transform(wave) + 1e-7 # fix zero values causing nans
    mel_spectrogram = t2(spec)
    # mel_spectrogram = torch.log(mel_spectrogram)
    mel_spectrogram = torch.log10(mel_spectrogram)
    mx = mel_spectrogram.max().item()
    mn = mel_spectrogram.min().item()
    mel_spectrogram = 2 * (mel_spectrogram - mn) / (mx - mn) - 1
    mel_spectrogram = F.pad(
        mel_spectrogram,
        (
            0,
            metadata.time_dim - mel_spectrogram.size(2),
            0,
            metadata.rescaled_height_dim - mel_spectrogram.size(1)
        ), 
        mode='constant', 
        value=0
    )
    return mel_spectrogram

def decode_spectrogram(spec, metadata):
    spec = spec[:, :metadata.n_mels, :metadata.vanilla_time_dim]
    dr = metadata.dynamic_range
    spec = (spec + 1) * (dr[1] - dr[0]) / 2 + dr[0]
    spec = torch.exp(spec)
    transform = torchaudio.transforms.InverseMelScale(
        metadata.time_dim // 2 + 1,
        n_mels=metadata.n_mels,
        sample_rate=metadata.sampling_rate
    )
    wave = torchaudio.functional.griffinlim(
        transform(spec),
        torch.hann_window(window_length=metadata.time_dim),
        metadata.time_dim,
        metadata.time_dim // 2,
        metadata.time_dim,
        2,
        128,
        0,
        metadata.sampling_rate * metadata.length_secs,
        True
    )
    return wave

def decode_spectrogram_diffwave(spec, metadata, args):
    model_dir = args.save_dir
    spec = spec[:, :metadata.n_mels, :metadata.vanilla_time_dim]
    audio, _ = diffwave_predict(spec, model_dir, fast_sampling=True)
    audio = audio.cpu().squeeze(0).numpy()
    slowed_audio = librosa.effects.time_stretch(audio, rate=0.5)
    slowed_audio_torch = torch.from_numpy(slowed_audio)
    return slowed_audio_torch.unsqueeze(0)