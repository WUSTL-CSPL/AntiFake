import torch
import torch.nn as nn
import yaml
from argparse import ArgumentParser
import pickle
import torch.nn.functional as F
import librosa
import numpy as np
import torchaudio

from adaptive_voice_conversion.model import AE, SpeakerEncoder, ContentEncoder, Decoder

class hp:
    '''Hyper parameters'''
    
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    data = "/data/private/voice/LJSpeech-1.0"
    # data = "/data/private/voice/nick"
    test_data = 'harvard_sentences.txt'
    max_duration = 10.0
    top_db = 15

    # signal processing
    sr = 24000 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 512 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 100 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/01"
    sampledir = 'samples'
    batch_size = 32
    
def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # print('fpath spect input')
    # print(y.shape)
    
    # Trimming
    y, _ = librosa.effects.trim(y, top_db=hp.top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    
    # print(mel.shape)

    return y, mel, mag

def get_spectrograms_tensor(y):
    
    # y = torch.cat((y[:1], y[1:] - 0.97 * y[:-1]))
    
    # Create MelSpectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sr, 
                                                           n_mels=hp.n_mels, 
                                                           n_fft=hp.n_fft, 
                                                           hop_length=hp.hop_length, 
                                                           win_length=hp.win_length,
                                                           norm = "slaney",
                                                           mel_scale = "slaney",
                                                           power = 1
                                                           ).cuda()

    # Compute Mel spectrogram
    mel = mel_spectrogram(y)
    mel = mel.squeeze(0)
    
    mel = 20 * torch.log10(torch.maximum(torch.tensor(1e-5), mel))
    mel = torch.clamp((mel - 20 + 100) / 100, 1e-8, 1)
    
    # print(mel.shape)

    return mel


def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

class AE_torch(nn.Module):
    def __init__(self, config):
        super(AE_torch, self).__init__()
        self.speaker_encoder = SpeakerEncoder(**config['SpeakerEncoder']) 
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, x):
        emb = self.speaker_encoder(x)
        mu, log_sigma = self.content_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        return mu, log_sigma, emb, dec

    def inference(self, x, x_cond):
        emb = self.speaker_encoder(x_cond)
        mu, _ = self.content_encoder(x)
        dec = self.decoder(mu, emb)
        return dec
    
    def get_embeddings(self, x, x_cond):
        emb = self.speaker_encoder(x_cond)
        mu, _ = self.content_encoder(x)
        return emb, mu

    def get_speaker_embeddings(self, x_cond):
        emb = self.speaker_encoder(x_cond)
        return emb
    
    def get_speaker_encoder(self):
        return self.speaker_encoder

    def get_speaker_embeddings(self, x):
        emb = self.speaker_encoder(x)
        return emb
    
class Inferencer(object):
    def __init__(self, config, original, target, args = None):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config

        self.args = args
        self.attr = './adaptive_voice_conversion/attr.pkl'
        self.model_path = './adaptive_voice_conversion/vctk_model.ckpt'
        self.original = original
        self.target = target

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        self.model.load_state_dict(torch.load(f'{self.model_path}'))
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        # print(self.model)
        self.model.eval()
        return
    
    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret
    
    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

def inference_one_utterance_torch(inferencer: Inferencer, x, x_cond):
        x = inferencer.utt_make_frames(x)
        x_cond = inferencer.utt_make_frames(x_cond)
        
        # print('x_cond')
        # print(x_cond.requires_grad)
        
        encoder_model = AE_torch(inferencer.config)
        encoder_model = encoder_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        encoder_model.eval()
        
        emb, mu = encoder_model.get_embeddings(x, x_cond)
        return emb, mu

def inference_from_path_torch(inferencer: Inferencer):
        _, src_mel, _ = get_spectrograms(inferencer.args.source)
        _, tar_mel, _ = get_spectrograms(inferencer.args.target)
        src_mel = torch.from_numpy(inferencer.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(inferencer.normalize(tar_mel)).cuda()
        tar_mel.requires_grad_()
        emb, mu = inference_one_utterance_torch(inferencer, src_mel, tar_mel)
        return src_mel, tar_mel, emb, mu

def extract_speaker_embedding_torch(inferencer: Inferencer):
    
        original_wav, original_mel, _ = get_spectrograms(inferencer.original)
        target_wav, target_mel, _ = get_spectrograms(inferencer.target)
        original_mel = torch.from_numpy(inferencer.normalize(original_mel)).cuda()
        target_mel = torch.from_numpy(inferencer.normalize(target_mel)).cuda()
        original_mel.requires_grad_()
        target_mel.requires_grad_()
        
        x_original = inferencer.utt_make_frames(original_mel)
        x_target = inferencer.utt_make_frames(target_mel)
        encoder_model = AE_torch(inferencer.config)
        encoder_model = encoder_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        encoder_model.eval()
        original_emb = encoder_model.get_speaker_embeddings(x_original)
        target_emb = encoder_model.get_speaker_embeddings(x_target)
        _model = encoder_model.get_speaker_encoder
        
        return original_wav, target_wav, original_mel, target_mel, original_emb, target_emb


    