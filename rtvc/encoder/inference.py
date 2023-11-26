from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav   # We want to expose this function from here
from encoder.audio import preprocess_wav_torch   # We want to expose this function from here
from matplotlib import cm
from encoder import audio
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
import torchaudio
import torch.nn.functional as F

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

_model = None # type: SpeakerEncoder
_device = None # type: torch.device


def load_model(weights_fpath: Path, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the
    first call to embed_frames() with the default weights file.

    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The
    model will be loaded and will run on this device. Outputs will however always be on the cpu.
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))


def is_loaded():
    return _model is not None


def embed_frames_batch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram.

    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed

def embed_frames_batch_torch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram.

    :param frames_batch: a batch mel of spectrogram as a PyTorch tensor of shape
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a PyTorch tensor of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = frames_batch.to(_device)
    embed = _model.forward(frames)
    return embed

def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to
    its spectrogram. This function assumes that the mel spectrogram parameters used are those
    defined in params_data.py.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
    then the last partial utterance will be considered, as if we padded the audio. Otherwise,
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
    utterances are entirely disjoint.
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
    respectively the waveform and the mel spectrogram with these slices to obtain the partial
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]        
    # # Exameple of the output
    # mel_slices:  [slice(0, 160, None), slice(80, 240, None)]
    # wav_slices:  [slice(0, 25600, None), slice(12800, 38400, None)]

    return wav_slices, mel_slices

def compute_partial_slices_torch(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to
    its spectrogram. This function assumes that the mel spectrogram parameters used are those
    defined in params_data.py.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
    then the last partial utterance will be considered, as if we padded the audio. Otherwise,
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
    utterances are entirely disjoint.
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
    respectively the waveform and the mel spectrogram with these slices to obtain the partial
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    n_samples = torch.tensor(n_samples)
    partial_utterance_n_frames = torch.tensor(partial_utterance_n_frames)
    overlap = torch.tensor(overlap)

    samples_per_frame = torch.tensor(int((sampling_rate * mel_window_step / 1000)))
    n_frames = int(torch.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(torch.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = torch.tensor([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]        
    # # Exameple of the output
    # mel_slices:  [slice(0, 160, None), slice(80, 240, None)]
    # wav_slices:  [slice(0, 25600, None), slice(12800, 38400, None)]

    return wav_slices, mel_slices

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.

    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their
    normalized average. If False, the utterance is instead computed from feeding the entire
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
    returned. If <using_partials> is simultaneously set to False, both these values will be None
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
                
    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    # print("frames_numpy.size: ", frames.size)
    # print("frames_numpy: ", frames)
    # # Show the original mel spectrogram
    # hop_length=int(sampling_rate * mel_window_step / 1000)
    # plot_mel_spectrogram(frames, sampling_rate, hop_length)
    
    # The following is for audio reconstruction test
    # wav_reconstructed = audio.mel_spectrogram_to_wav_and_save(frames)
    # sf.write('reconstructed_audio.wav', wav_reconstructed, sampling_rate)
    
    ###### This part was substituted by the following code ######
    frames_batch = np.array([frames[s] for s in mel_slices])
    frames = torch.from_numpy(frames_batch).to(_device)
    partial_embeds = _model.forward(frames).detach().cpu().numpy()
    ###### This part was substituted by the following code ######    
     
    # embeds_list = []
    # for s in mel_slices:
    #     # # Show the mel spectrogram of slices
    #     # plot_mel_spectrogram(frames[s], sampling_rate, hop_length)
    #     # wave = mel_spectrogram_to_wav(frames[s])
        
    #     frame_tensor = torch.from_numpy(frames[s]).unsqueeze(0).to(_device)
    #     # Input shape: (batch_size, n_mels, n_frames) -> (1, 160, 40); Output shape: (1, 256)
    #     frame_tensor.requires_grad = True
    #     _model.train()
    #     print("frame_tensor.shape: ", frame_tensor.shape)
    #     print("frame_tensor: ", frame_tensor)
    #     embed = _model.forward(frame_tensor)
    #     embeds_sum = embed.sum()
    #     embeds_sum.backward()
    #     input_grad = frame_tensor.grad
    #     print("input_grad.shape: ", input_grad.shape)
    #     print("input_grad: ", input_grad)
    #     _model.eval()
    #     embed = embed.detach().cpu().numpy()
    #     embeds_list.append(embed)
    # partial_embeds = np.concatenate(embeds_list, axis=0)
    
    # Compute the utterance embedding from the partial embeddings
    # The shape of raw_embed is 256
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

def embed_utterance_preprocess(wav, using_partials=True, **kwargs):

    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        
        print("Using the entire utterance, please change the code to use partials.")

        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    # print("len(wav): ", len(wav))
    # print("wave_slices: ", wave_slices)
    # print("mel_slices: ", mel_slices)
    # exit(0)
    # wave_slices, mel_slices = compute_partial_slices_torch(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        # wav = torch.cat((wav, torch.zeros(max_wave_length - len(wav))), 0)
        
    # Original frames in the numpy asrray
    # frames = audio.wav_to_mel_spectrogram(wav)
    # frames_sum = frames.sum()
    # print("frames.shape: ", frames.shape)
    # print("frames_tensor: ", frames)
    # print("frames_sum: ", frames_sum)
    # exit(0)
    
    # # Show the original mel spectrogram in numpy
    # hop_length=int(sampling_rate * mel_window_step / 1000)
    # plot_mel_spectrogram(frames, sampling_rate, hop_length)
    # exit(0)
    
    
    # # Convert wav to tensor
    # wav_tensor = torch.from_numpy(wav).to(_device)
    # wav_tensor.requires_grad = True
    
    # frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor)
    # frames_tensor_sum = frames_tensor.sum()
    # print("frames_tensor.shape: ", frames_tensor.shape)
    # print("frames_tensor: ", frames_tensor)
    # print("frames_tensor_sum: ", frames_tensor_sum)
    # exit(0)
    
    # # The following is for audio reconstruction test
    # wav_reconstructed = audio.mel_spectrogram_to_wav(frames)
    # print("wav_reconstructed.shape: ", wav_reconstructed.shape)
    # print("wav_reconstructed: ", wav_reconstructed)
    # sf.write('reconstructed_audio.wav', wav_reconstructed, sampling_rate)
    
    ###### This part was substituted by the following code ######
    # frames_batch = np.array([frames[s] for s in mel_slices])
    # frames = torch.from_numpy(frames_batch).to(_device)
    # partial_embeds = _model.forward(frames).detach().cpu().numpy()
    ###### This part was substituted by the following code ######    
     
    # embeds_list = []
    # frame_tensor_list = []
    # for s in mel_slices:
    #     # # Show the mel spectrogram of slices
    #     # plot_mel_spectrogram(frames[s], sampling_rate, hop_length)
    #     # wave = mel_spectrogram_to_wav(frames[s])

    #     frame_tensor = torch.from_numpy(frames[s]).unsqueeze(0).to(_device)
    #     # frame_tensor = frames_tensor[:, s].unsqueeze(0).to(_device)
    #     # print("frame_tensor.shape: ", frame_tensor.shape)
    #     frame_tensor.requires_grad = True
    #     frame_tensor_list.append(frame_tensor)
    #     # Input shape: (batch_size, n_mels, n_frames) -> (1, 160, 40); Output shape: (1, 256)
        
    #     _model.train()
    #     embed = _model.forward(frame_tensor)
    #     # embeds_sum = embed.sum()
    #     # embeds_sum.backward()
    #     input_grad = frame_tensor.grad
    #     # print("input_grad.shape: ", input_grad.shape)
    #     # print("input_grad: ", input_grad)
    #     _model.eval()
        
    #     embeds_list.append(embed)

    # return embeds_list, frame_tensor_list
    # return frames, mel_slices, _model, _device
    return wav, wave_slices, mel_slices, _model, _device

def plot_attributions(attributions, cmap="viridis"):
    fig, ax = plt.subplots()
    im = ax.imshow(attributions.squeeze().numpy(), cmap=cmap, aspect='auto', origin='lower')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel frequency')
    plt.colorbar(im, ax=ax, label='Attribution')
    plt.show()

def embed_utterance_torch(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.

    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a PyTorch tensor of shape (num_channels, num_samples)
    :param using_partials: if True, then the utterance is split in partial utterances of
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their
    normalized average. If False, the utterance is instead computed from feeding the entire
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a PyTorch tensor of shape (model_embedding_size,). If
    <return_partials> is True, the partial utterances as a PyTorch tensor of shape
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
    returned. If <using_partials> is simultaneously set to False, both these values will be None
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram_torch(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices_torch(wav.shape[1], **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= wav.shape[1]:
        wav = F.pad(wav, (0, max_wave_length - wav.shape[1]), "constant")

    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram_torch(wav)
    frames_batch = torch.stack([frames[:, s] for s in mel_slices])
    partial_embeds = embed_frames_batch_torch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = torch.mean(partial_embeds, dim=0)
    embed = raw_embed / torch.norm(raw_embed, p=2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplemented()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)


def plot_mel_spectrogram(frames, sampling_rate, hop_length):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(frames.T, ref=np.max), sr=sampling_rate, hop_length=hop_length, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()