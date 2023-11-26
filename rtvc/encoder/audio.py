from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import soundfile as sf
import struct
import torch
import torchaudio
import torch.nn.functional as F

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav))
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(y=wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
        
    # sf.write("preprocessed_audio.wav", wav, sampling_rate)
    
    return wav

def preprocess_wav_torch(fpath_or_wav: Union[str, Path, torch.Tensor], 
                   source_sr: Optional[int] = None, 
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = False):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a PyTorch tensor of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = torchaudio.load(fpath_or_wav)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(source_sr, sampling_rate)
        wav = resampler(wav)
    
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume_torch(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences_torch(wav)
    
    return wav

def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

def wav_to_mel_spectrogram_torch(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    # wav = wav.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
        norm = "slaney",
        mel_scale = "slaney"
    ).to(wav.device)
    frames = mel_transform(wav).squeeze(0)
    return frames.transpose(0, 1).float()
    
def mel_spectrogram_to_wav(mel_spectrogram):
    mel_spectrogram = mel_spectrogram.T
    reconstructed_wav = librosa.feature.inverse.mel_to_audio(mel_spectrogram,
                                               n_fft=int(sampling_rate * mel_window_length / 1000),
                                               hop_length=int(sampling_rate * mel_window_step / 1000),
                                               sr=sampling_rate)
    
    return reconstructed_wav

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def trim_long_silences_torch(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a PyTorch tensor of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    remainder = wav.shape[0] % samples_per_window
    if remainder != 0:
        wav = F.pad(wav, (0, samples_per_window - remainder), 'constant', 0)
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = torch.clamp(torch.round(wav * int16_max), -int16_max, int16_max - 1).to(torch.int16)
    pcm_wave = pcm_wave.flatten().numpy().tobytes()
    
    # Perform voice activation detection
    voice_flags = []
    vad = torchaudio.transforms.Vad(sample_rate=sampling_rate)
    for window_start in range(0, len(pcm_wave), samples_per_window*2):
        window_end = window_start + samples_per_window*2
        voice_flags.append(vad(pcm_wave[window_start:window_end]))
    voice_flags = torch.tensor(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = F.pad(array, (width//2, width//2), mode='constant', value=0)
        kernel = torch.ones(width)/width
        return F.conv1d(array_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=width//2).squeeze(0)
    
    audio_mask = moving_average(voice_flags.float(), vad_moving_average_width)
    audio_mask = torch.round(audio_mask).bool()
    
    # Dilate the voiced regions
    audio_mask = audio_mask.unsqueeze(-1).repeat(1, samples_per_window).flatten()
    audio_mask = torch.nn.functional.max_pool1d(audio_mask.unsqueeze(0).unsqueeze(0), vad_max_silence_length*2+1, stride=1, padding=vad_max_silence_length).squeeze().bool()
    audio_mask = audio_mask.unsqueeze(-1).repeat(1, samples_per_window).flatten()
    
    return wav[audio_mask]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def normalize_volume_torch(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = torch.tensor(target_dBFS) - 10 * torch.log10(torch.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * torch.pow(10, dBFS_change / 20)