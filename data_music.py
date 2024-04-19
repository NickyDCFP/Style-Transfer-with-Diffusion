from easydict import EasyDict
from collections import OrderedDict
import torch
import torchaudio.transforms
from torchaudio.datasets import GTZAN
import torch.nn.functional as F
from diffwave.inference import predict as diffwave_predict

def get_metadata() -> EasyDict:
    return EasyDict({
        "sampling_rate" : 22050,
        "length_secs" : 30,
        "num_channels" : 1,
        "time_dim" : 1024,
        "n_mels" : 80,
        "rescaled_height_dim" : 128,
        "vanilla_time_dim" : 862,
        "dynamic_range" : [-16, 8],
        "hop_samples" : 256,
    })
    
def get_gtzan(data_dir: str):
    dataset = GTZAN(data_dir, download=False)
    return dataset

def collate_gtzan(minibatch):
    metadata = get_metadata()
    length_samples = metadata.sampling_rate * metadata.length_secs
    return [
        sample[0][:, :length_samples]
        for sample in minibatch if sample[1] == metadata.sampling_rate
    ]

def encode_spectrogram(wave, metadata):
  wave = torch.clamp(wave[0], -1.0, 1.0)
  mel_args = {
      'sample_rate': metadata.sampling_rate,
      'win_length': metadata.hop_samples * 4,
      'hop_length': metadata.hop_samples,
      'n_fft': 1024, #metadata.time_dim,
      'f_min': 20.0,
      'f_max': metadata.sampling_rate / 2.0,
      'n_mels': metadata.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(wave)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    spectrogram = spectrogram[:, :metadata.time_dim]
    spectrogram = F.pad(spectrogram, (0, 0, 0, metadata.rescaled_height_dim - spectrogram.size(0)))

    return spectrogram.unsqueeze(0)
  
def decode_spectrogram(spec, metadata, args = None):
    model_dir = args.save_dir if args is not None else './trained_models/'
    spec = spec[:, :metadata.n_mels, :]
    audio, _ = diffwave_predict(spec, model_dir, fast_sampling=True)
    return audio.cpu()

def decode_spectrogram_griffinlim(spec, metadata):
    spec = spec[:, :metadata.n_mels, :metadata.vanilla_time_dim]
    dr = metadata.dynamic_range
    spec = (spec + 1) * (dr[1] - dr[0]) / 2 + dr[0]
    spec = torch.exp(spec) # power 10 must be
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

if __name__ == "__main__":
    w = get_gtzan('./dataset/GTZAN')[0][0]
    metadata = get_metadata()
    spec = encode_spectrogram(w, metadata)
    torchaudio.save(
        'test.wav',
        decode_spectrogram(spec, metadata),
        22050
    )