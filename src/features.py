
import numpy as np
import librosa
import torch
import torchaudio

# Paramètres audio par défaut
SR = 16000
N_MELS = 64
HOP_LENGTH = 160   # 10 ms @16k
N_FFT = 400        # 25 ms

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


def load_mono_resample(path, sr=SR, duration=4.0):
    y, s = librosa.load(path, sr=None, mono=True)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    target = int(sr*duration)
    if len(y) >= target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target-len(y)))
    return y.astype(np.float32)

def wav_to_logmels(y):
    y_t = torch.from_numpy(y)
    S = mel_transform(y_t)
    S_db = amplitude_to_db(S+1e-9)
    S_db = (S_db - S_db.mean()) / (S_db.std()+1e-5)
    return S_db  # [n_mels, T]

def extract_mfcc_stats(path, sr=SR, n_mfcc=13):
    y, s = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
