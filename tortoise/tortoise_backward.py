import torch
import torchaudio
import torch.nn as nn
import librosa
import os
import math
from einops import rearrange
import functools
from transformers import GPT2Config, GPT2Model
from abc import abstractmethod
import random
import torch.nn.functional as F
from glob import glob
from scipy.io.wavfile import read
import numpy as np
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
import librosa.util as librosa_util
from librosa.filters import mel as librosa_mel_fn


DEFAULT_MEL_NORM_FILE='./tortoise/mel_norms.pth'
BUILTIN_VOICES_DIR = './tortoise/voices'
AUTOREGRESSIVE_ENCODER = './tortoise/autoregressive.pth'
DIFFUSION_ENCODER = './tortoise/diffusion_decoder.pth'
TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254

def get_voices(extra_voice_dirs=[]):
    dirs = [BUILTIN_VOICES_DIR] + extra_voice_dirs
    voices = {}
    for d in dirs:
        subs = os.listdir(d)
        for sub in subs:
            subj = os.path.join(d, sub)
            if os.path.isdir(subj):
                voices[sub] = list(glob(f'{subj}/*.wav')) + list(glob(f'{subj}/*.mp3')) + list(glob(f'{subj}/*.pth'))
    return voices

def load_voice(voice, extra_voice_dirs=[]):
    if voice == 'random':
        return None, None

    voices = get_voices(extra_voice_dirs)
    paths = voices[voice]
    if len(paths) == 1 and paths[0].endswith('.pth'):
        return None, torch.load(paths[0])
    else:
        conds = []
        for cond_path in paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        return conds, None

def load_voice_path(path):
    conds = []
    c = load_audio(path, 22050)
    conds.append(c)
    return conds

def wav_to_univnet_mel(wav, do_normalization=False, device='cuda'):
    # print('1')
    # print(wav.requires_grad)
    stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
    stft = stft.to(device)
    mel = stft.mel_spectrogram(wav)
    # print('2')
    # print(mel.requires_grad)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)
    
    # print('3')
    # print(mel.requires_grad)
    return mel

def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -10)
        assert(torch.max(y.data) <= 10)
        y = torch.clip(y, min=-1, max=1)

        magnitudes, phases = self.stft_fn.transform(y)
        # magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == '.mp3':
        audio, lsr = librosa.load(audiopath, sr=sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        assert False, f"Unsupported audio format provided: {audiopath[-4:]}"

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)

def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]

class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        # print(inp.shape)
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


def format_conditioning(clip, cond_length=132300, device='cuda'):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    # print('clip shape here')
    # print(clip.shape)
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)

# def format_conditioning_single(clip, cond_length=132300, device='cuda'):
#     """
#     Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
#     """
#     # print('clip shape here')
#     # print(clip.shape)
#     gap = clip.shape[-1] - cond_length
#     if gap < 0:
#         clip = F.pad(clip, pad=(0, abs(gap)))
#     elif gap > 0:
#         rand_start = random.randint(0, gap)
#         clip = clip[:, rand_start:rand_start + cond_length]
#     mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
#     return mel_clip.unsqueeze(0).to(device)

def get_conditioning_latents_torch(tts, voice_samples, return_mels=False):
    """
    Transforms one or more voice_samples into a tuple (autoregressive_conditioning_latent, diffusion_conditioning_latent).
    These are expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic
    properties.
    :param voice_samples: List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
    """
    voice_samples = [v.to(tts.device) for v in voice_samples]

    auto_conds = []
    if not isinstance(voice_samples, list):
        voice_samples = [voice_samples]
    for vs in voice_samples:
        # torchvision mel spectrogram
        auto_conds.append(format_conditioning(vs, device=tts.device))
    auto_conds = torch.stack(auto_conds, dim=1)
    tts.autoregressive = tts.autoregressive.to(tts.device)
    # Transformer inside UnifiedVoice custom model to get conditioning
    auto_latent = tts.autoregressive.get_conditioning(auto_conds)

    diffusion_conds = []
    for sample in voice_samples:
        # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
        sample = torchaudio.functional.resample(sample, 22050, 24000)
        sample = pad_or_truncate(sample, 102400)
        cond_mel = wav_to_univnet_mel(sample.to(
            tts.device), do_normalization=False, device=tts.device)  # tacotron mel spectrum encoder
        diffusion_conds.append(cond_mel)
    diffusion_conds = torch.stack(diffusion_conds, dim=1)

    tts.diffusion = tts.diffusion.to(tts.device)
    diffusion_latent = tts.diffusion.get_conditioning(
        diffusion_conds)  # custom contextual embedder model

    if return_mels:
        return auto_latent, diffusion_latent, auto_conds, diffusion_conds
    else:
        return auto_latent, diffusion_latent

def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                             n_positions=max_mel_seq_len+max_text_seq_len,
                             n_ctx=max_mel_seq_len+max_text_seq_len,
                             n_embd=model_dim,
                             n_layer=layers,
                             n_head=heads,
                             gradient_checkpointing=checkpointing,
                             use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim),\
           None, None

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class TextToSpeech:
    def __init__(self):
        self.device = torch.device('cuda')
        self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                        model_dim=1024,
                                        heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                        train_solo_embeddings=False).cuda().eval()
        self.autoregressive.load_state_dict(torch.load(AUTOREGRESSIVE_ENCODER))
        self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                        in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                        layer_drop=0, unconditioned_percentage=0).cuda().eval()
        self.diffusion.load_state_dict(torch.load(DIFFUSION_ENCODER))


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=None, number_mel_codes=8194, start_mel_token=8192,
                 stop_mel_token=8193, train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
        """
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = number_text_tokens * types if start_text_token is None else start_text_token
        self.stop_text_token = 0
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads).cuda()
        self.text_embedding = nn.Embedding(self.number_text_tokens*types+1, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens+2+self.max_conditioning_inputs, self.max_text_tokens+2, checkpointing)
        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens*types+1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = speech_conditioning_input.unsqueeze(1) if len(
            speech_conditioning_input.shape) == 3 else speech_conditioning_input
        conds = []
        # print('total speech_conditioning_input')
        # print(speech_conditioning_input.shape)
        for j in range(speech_conditioning_input.shape[1]):
            # print(self.conditioning_encoder(speech_conditioning_input[:, j]).shape)
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        # print(conds.shape)
        return conds
    
class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        # print('auto input shape')
        # print(x.shape)
        # print(x.shape)
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            # print(h.mean(dim=2).shape)
            return h.mean(dim=2)
        else:
            # print(h[:, :, 0].shape)
            return h[:, :, 0]
        

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)
    

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)

class DiffusionTts(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            in_latent_channels=512,
            in_tokens=8193,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = nn.Conv1d(in_channels, model_channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_embedding = nn.Embedding(in_tokens, model_channels)
        self.code_converter = nn.Sequential(
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.code_norm = normalization(model_channels)
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                 nn.Conv1d(model_channels, model_channels*2,3,padding=1,stride=2),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False))
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )

        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [ResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 3, padding=1),
        )

    def get_conditioning(self, conditioning_input):
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            # print('diffusion input shape')
            # print(speech_conditioning_input[:, j].shape)
            conds.append(self.contextual_embedder(speech_conditioning_input[:, j]))
        # print(conds.shape)
        conds = torch.cat(conds, dim=-1)
        # print(conds.shape)
        conds = conds.mean(dim=-1)
        # print('diffusion native')
        # print(conds.shape)
        return conds

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

if __name__ == "__main__":
    
    tts = TextToSpeech()

    # target speaker clips stored under tortoise-tts/tortoise/voices/<speaker>/
    voice = 'p333-emb'

    voice_samples, _ = load_voice(voice)
    
    for item in voice_samples:
        item.requires_grad_()
    
    auto_conditioning, diffusion_conditioning, auto_conds, _ = get_conditioning_latents_torch(
        tts, voice_samples, return_mels=True)
    
    print(auto_conds.shape)
    # print(voice_samples[0].shape)
    print(auto_conditioning.shape)
    print(diffusion_conditioning.shape)
    
    # ## test autoregresive backward to get grad
    # loss = auto_conditioning.sum()
    # loss.backward()
    # grad = voice_samples[0].grad
    # print(grad.shape)
    # np.savetxt('autoregressive_grad.txt', grad.numpy())
    
    # ## test diffusion backward to get grad
    # loss = diffusion_conditioning.sum()
    # loss.backward()
    # grad = voice_samples[0].grad
    # print(grad.shape)
    # np.savetxt('diffusion_grad.txt', grad.numpy())