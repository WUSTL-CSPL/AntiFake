from __future__ import print_function, division

import numpy as np
import os
from pathlib import Path
import time
import yaml
import torch
import numpy as np
import soundfile as sf
import sys
import io
import torch.nn as nn
import torchaudio
import csv
import random
import pygame

sys.path.insert(0, "./rtvc")
from encoder import inference as encoder
from encoder import audio
from encoder.params_data import *
from utils.default_models import ensure_default_models

sys.path.insert(0, "./TTS")
from TTS.api import TTS

from adaptive_voice_conversion.adaptivevc_backward import Inferencer, extract_speaker_embedding_torch, get_spectrograms_tensor
from adaptive_voice_conversion.model import SpeakerEncoder

from tortoise.tortoise_backward import load_voice_path, TextToSpeech, get_conditioning_latents_torch, format_conditioning, ConditioningEncoder, pad_or_truncate, wav_to_univnet_mel, AttentionBlock


############################################### Options ##################################################
OUTPUT_DIR = './output/protected.wav'

RTVC_LOSS = True
AVC_LOSS = True
COQUI_LOSS = True
TORTOISE_AUTOREGRESSIVE_LOSS = False
TORTOISE_DIFFUSION_LOSS = False

THRESHOLD_BASE = False
############################################### Configs ##################################################
TARGET_SPEAKER_DATABASE = './speakers_database'
NUM_RANDOM_TARGET_SPEAKER = 24 

RTVC_DEFAULT_MODEL_PATH = "./saved_models"
AVC_CONFIG_PATH = "./adaptive_voice_conversion/config.yaml"
COQUI_YOURTTS_PATH = "tts_models/multilingual/multi-dataset/your_tts"

SOURCE_SPEAKER_PATH = None 

AVC_ENCODER_MODEL = None
COQUI_ENCODER_MODEL = None
TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE = None
TORTOISE_ENCODER_MODEL_DIFFUSION = None
RTVC_ENCODER_MODEL = None

SAMPLING_RATE = 16000
ATTACK_ITERATIONS = 1000 
DEVICE = 'cuda'
######################################### Tunable Parameters #############################################
quality_weight_snr = 0.005
quality_weight_L2 = 0.05
quality_weight_frequency = 0.3 
learning_rate = 0.02
weight_decay_iter = 100 
weight_decay_rate = 0.9
avc_scale = 0.18
coqui_scale = 0.85
tortoise_autoregressive_scale = 0.02
tortoise_diffusion_scale = 0.014
rtvc_scale = 1
##########################################################################################################


def avc_loss(wav_tensor_updated, avc_embed_initial, avc_embed_target, avc_embed_threshold):
    wav_tensor_updated = torchaudio.functional.resample(wav_tensor_updated, SAMPLING_RATE, 24000)
    
    frames_tensor = get_spectrograms_tensor(wav_tensor_updated)   
    
    # Recompute the embeddings for the updated frame_tensor_list 
    frame_tensor = frames_tensor.unsqueeze(0).to(DEVICE)
    AVC_ENCODER_MODEL.train()
    embed = AVC_ENCODER_MODEL.forward(frame_tensor) 
    
    if THRESHOLD_BASE:
        elu = torch.nn.ELU()
        delta_L2 = elu(avc_embed_threshold - torch.norm(embed - avc_embed_initial, p=2)) * avc_scale 
    else:
        delta_L2 = torch.norm(embed - avc_embed_target, p=2) * avc_scale
    
    return delta_L2

def coqui_loss(wav_tensor_updated, coqui_embed_initial, coqui_embed_target, coqui_embed_threshold):
    wav_tensor_updated = torchaudio.functional.resample(wav_tensor_updated, SAMPLING_RATE, 16000)
    
    embed = COQUI_ENCODER_MODEL.encoder.compute_embedding(wav_tensor_updated)
    
    if THRESHOLD_BASE:
        elu = torch.nn.ELU()
        delta_L2 = elu(coqui_embed_threshold - torch.norm(embed - coqui_embed_initial, p=2)) * coqui_scale 
    else:
        delta_L2 = torch.norm(embed - coqui_embed_target, p=2) * coqui_scale
    
    return delta_L2

def tortoise_autoregressive_loss(wav_tensor_updated, tortoise_source_emb_autoregressive, tortoise_target_emb_autoregressive, tortoise_threshold_autoregressive):
    wav_tensor_updated = torchaudio.functional.resample(wav_tensor_updated, SAMPLING_RATE, 22050)
    
    frames_tensor_autoregressive = format_conditioning(wav_tensor_updated).to(DEVICE)  
    frames_tensor_autoregressive = frames_tensor_autoregressive.unsqueeze(0).to(DEVICE)
    
    TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE.train()
    embed_autoregressive = TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE.forward(frames_tensor_autoregressive[0])  
    
    if THRESHOLD_BASE:
        elu = torch.nn.ELU()
        delta_L2 = elu(tortoise_threshold_autoregressive - torch.norm(embed_autoregressive - tortoise_source_emb_autoregressive, p=2)) * tortoise_autoregressive_scale * 50
    else:
        delta_L2 = torch.norm(embed_autoregressive - tortoise_target_emb_autoregressive, p=2) * tortoise_autoregressive_scale
    
    return delta_L2

def tortoise_diffusion_loss(wav_tensor_updated, tortoise_source_emb_diffusion, tortoise_target_emb_diffusion, tortoise_threshold_diffusion):
    wav_tensor_updated = torchaudio.functional.resample(wav_tensor_updated, SAMPLING_RATE, 24000)
    
    wav_tensor_updated = pad_or_truncate(wav_tensor_updated, 102400)
    frames_tensor_diffusion = wav_to_univnet_mel(wav_tensor_updated.to(DEVICE), do_normalization=False, device=DEVICE)  
    frames_tensor_diffusion = frames_tensor_diffusion.unsqueeze(0).to(DEVICE)
    
    TORTOISE_ENCODER_MODEL_DIFFUSION.train()
    embed_diffusion = TORTOISE_ENCODER_MODEL_DIFFUSION.forward(frames_tensor_diffusion[0])     
    embed_diffusion = embed_diffusion.mean(dim=-1)    
    
    if THRESHOLD_BASE:
        elu = torch.nn.ELU()
        delta_L2 = elu(tortoise_threshold_diffusion - torch.norm(embed_diffusion - tortoise_source_emb_diffusion, p=2)) * tortoise_diffusion_scale * 50
    else:
        delta_L2 = torch.norm(embed_diffusion - tortoise_target_emb_diffusion, p=2) * tortoise_diffusion_scale
        
    return delta_L2

def rtvc_loss(wav_tensor_updated, rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list, rtvc_embed_initial, rtvc_embed_target, rtvc_embed_threshold):
    frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_updated).to(DEVICE)
    delta_L2_total = 0
    # Recompute the embeddings for the updated frame_tensor_list
    for i, s in enumerate(rtvc_mel_slices):
        frame_tensor = frames_tensor[s].unsqueeze(0).to(DEVICE)
        rtvc_frame_tensor_list[i] = frame_tensor
        
        RTVC_ENCODER_MODEL.train()
        embed = RTVC_ENCODER_MODEL.forward(frame_tensor)       
        rtvc_embeds_list[i] = embed
    
    if THRESHOLD_BASE:
        elu = torch.nn.ELU()
        for i, frame_tensor in enumerate(rtvc_frame_tensor_list):
            delta_L2 = torch.norm(rtvc_embeds_list[i] - rtvc_embed_initial, p=2) * rtvc_scale 
            delta_L2_total += delta_L2
        delta_L2_total = elu(rtvc_embed_threshold - delta_L2_total) 
    else:
        for i, frame_tensor in enumerate(rtvc_frame_tensor_list):
            delta_L2 = torch.norm(rtvc_embeds_list[i] - rtvc_embed_target, p=2) * rtvc_scale
            delta_L2_total += delta_L2         
    
    return delta_L2_total

def frequency_filter(wav_diff):
    # get spectrogram
    spectrogram = torchaudio.transforms.Spectrogram().cuda()
    diff_spec = spectrogram(wav_diff)[0]

    # load csv
    xs = []
    ys = []
    with open('./points.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))

    # ys is scaled to 0-1 inversely, with originally large values close to 0, vice versa
    ys_scaled = [1 - (item + 20) / 100 for item in ys]
    ys = ys_scaled

    # for each 201 windows, 201 bc fft window is defaulted to 400
    for i in range(0, diff_spec.shape[0]): 
        # by the nyquist theorem, signal processing can only reach half of the sampling rate
        bin_freq = SAMPLING_RATE / 2 / 200 
        
        # middle point at each bin
        probe_freq = (i + 0.5) * bin_freq
        
        # use linear interpolation
        for j, x in enumerate(xs):
            if xs[j] < probe_freq and xs[j + 1] > probe_freq:
                weight_freq = ys[j] + ((probe_freq - xs[j]) * (ys[j + 1] - ys[j])) / (xs[j + 1] - xs[j])
        
        diff_spec[i] *= weight_freq

    # sum up the loss, divide by the length
    loss = torch.sum(diff_spec) / len(diff_spec) 
    return loss

def attack_iteration(wav_tensor_list, 
                    avc_embed_initial = None,
                    avc_embed_target = None,
                    avc_embed_threshold = None,
                    coqui_embed_initial = None,
                    coqui_embed_target = None,
                    coqui_embed_threshold = None,
                    tortoise_source_emb_autoregressive = None,
                    tortoise_target_emb_autoregressive = None,
                    tortoise_threshold_autoregressive = None,
                    tortoise_source_emb_diffusion = None,
                    tortoise_target_emb_diffusion = None,
                    tortoise_threshold_diffusion = None,
                    rtvc_mel_slices = None,
                    rtvc_embeds_list = None,
                    rtvc_frame_tensor_list = None,
                    rtvc_embed_initial = None,
                    rtvc_embed_target = None,
                    rtvc_embed_threshold = None,
                    ):
    
    start_time = time.time()

    global learning_rate
    
    for iter in range(ATTACK_ITERATIONS):
        
        if iter % (weight_decay_iter) == 0 and iter != 0:
            learning_rate = learning_rate * weight_decay_rate
            
        loss = 0
        wav_tensor_updated = wav_tensor_list[0]
        
        # increment loss for each encoder 
        if AVC_LOSS:
            avc_delta_L2 = avc_loss(wav_tensor_updated, avc_embed_initial, avc_embed_target, avc_embed_threshold)
            loss += avc_delta_L2
        
        if COQUI_LOSS:
            coqui_delta_L2 = coqui_loss(wav_tensor_updated, coqui_embed_initial, coqui_embed_target, coqui_embed_threshold)
            loss += coqui_delta_L2
        
        if TORTOISE_AUTOREGRESSIVE_LOSS:
            delta_L2_autoregressive = tortoise_autoregressive_loss(wav_tensor_updated, tortoise_source_emb_autoregressive, tortoise_target_emb_autoregressive, tortoise_threshold_autoregressive)
            loss += delta_L2_autoregressive
            
        if TORTOISE_DIFFUSION_LOSS:
            delta_L2_diffusion = tortoise_diffusion_loss(wav_tensor_updated, tortoise_source_emb_diffusion, tortoise_target_emb_diffusion, tortoise_threshold_diffusion)
            loss += delta_L2_diffusion
        
        if RTVC_LOSS:
            delta_L2_rtvc = rtvc_loss(wav_tensor_updated, rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list, rtvc_embed_initial, rtvc_embed_target, rtvc_embed_threshold)
            loss += delta_L2_rtvc
        
        # calculate quality norm
        quality_l2_norm = torch.norm(wav_tensor_updated - wav_tensor_initial, p=2)
        
        # calculate snr
        diff_waveform_squared = torch.square(wav_tensor_updated - wav_tensor_initial)
        signal_power = torch.mean(torch.square(wav_tensor_updated))
        noise_power = torch.mean(diff_waveform_squared)
        quality_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)) 
        
        # calculate frequency filter
        quality_frequency = frequency_filter(wav_tensor_updated - wav_tensor_initial)
        
        # aggregate loss 
        quality_term = quality_weight_snr * quality_snr - quality_weight_L2 * quality_l2_norm - quality_weight_frequency * quality_frequency
        loss = - loss - quality_term

        print("Quality term: ", quality_term)
        print("Loss: ", loss)
            
        loss.backward(retain_graph=True)
        
        attributions = wav_tensor_updated.grad.data
        
        with torch.no_grad():
            
            mean_attributions = torch.mean(torch.abs(attributions))
            # print("Attributions_mean: ", mean_attributions)
            sign_attributions = torch.sign(attributions)
            wav_tensor_updated_clone = wav_tensor_updated.clone()
            wav_tensor_updated_clone += learning_rate * sign_attributions
            
            # Clip the values of the wav_tensor_updated_clone by using tanh function
            wav_tensor_updated_clone = torch.clamp(wav_tensor_updated_clone, -1, 1)
            
            wav_tensor_list[0] = wav_tensor_updated_clone
            wav_tensor_list[0].requires_grad = True
            # Clear gradients for the next iteration
            wav_tensor_updated.grad.zero_()   
        

        if iter == ATTACK_ITERATIONS - 1:
            wav_updated = wav_tensor_updated.detach().cpu().numpy().squeeze()
            sf.write(OUTPUT_DIR, wav_updated, SAMPLING_RATE)
        
        # Calculate the progress of the attack
        progress = (iter + 1) / ATTACK_ITERATIONS
        
        # Update the progress bar
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress:.2%}', end='', flush=True)
        print("\n")
    
    end_time = time.time()

    used_time = end_time - start_time

    # Print the optimization time in hours, minutes and seconds
    print("Time used: %d hours, %d minutes, %d seconds" % (used_time // 3600, (used_time % 3600) // 60, used_time % 60))

# Compute embedding with RTVC 
def rtvc_embed(wav_tensor_initial, mel_slices, target_speaker_path):

    embeds_list = []
    frame_tensor_list = []
    frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_initial).to(DEVICE)
    
    # Get source embeddings
    for s in mel_slices:
        frame_tensor = frames_tensor[s].unsqueeze(0).to(DEVICE)
        frame_tensor_list.append(frame_tensor)
        RTVC_ENCODER_MODEL.train()
        embed = RTVC_ENCODER_MODEL.forward(frame_tensor)
        embeds_list.append(embed)

    partial_embeds = torch.stack(embeds_list, dim=0)
    raw_embed_initial = torch.mean(partial_embeds, dim=0, keepdim=True)

    # Get target embeddings
    preprocessed_wav_target = encoder.preprocess_wav(target_speaker_path, SAMPLING_RATE)
    wav_target, _, _, _, _ = encoder.embed_utterance_preprocess(preprocessed_wav_target, using_partials=True)

    wav_tensor_target = torch.from_numpy(wav_target).unsqueeze(0).to(DEVICE)
    frames_tensor_target = audio.wav_to_mel_spectrogram_torch(wav_tensor_target).to(DEVICE)
    embeds_list_target = []
        
    for s in mel_slices:
        try:
            frame_tensor_target = frames_tensor_target[s].unsqueeze(0).to(DEVICE)
            embed_target = RTVC_ENCODER_MODEL.forward(frame_tensor_target) 
            embeds_list_target.append(embed_target)
        except:
            pass

    partial_embeds_target = torch.stack(embeds_list_target, dim=0)
    raw_embed_target = torch.mean(partial_embeds_target, dim=0, keepdim=True)

    return mel_slices, frame_tensor_list, embeds_list, raw_embed_initial, raw_embed_target

# Compute embedding with RTVC 
def avc_embed(source_speaker_path, target_speaker_path):
    with open(AVC_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    inferencer = Inferencer(config=config, original = source_speaker_path, target = target_speaker_path)
    _, _, _, _, avc_initial_emb, avc_target_emb = extract_speaker_embedding_torch(inferencer)
    global AVC_ENCODER_MODEL 
    AVC_ENCODER_MODEL = SpeakerEncoder(**inferencer.config['SpeakerEncoder']).cuda()
    return avc_initial_emb, avc_target_emb

# Compute embedding with COQUI 
def coqui_embed(source_speaker_path, target_speaker_path):
    null_stream = io.StringIO() 
    sys.stdout = null_stream
    tts = TTS(model_name=COQUI_YOURTTS_PATH, progress_bar=True, gpu=True)
    speaker_manager = tts.synthesizer.tts_model.speaker_manager
    source_wav = speaker_manager.encoder_ap.load_wav(source_speaker_path, sr=speaker_manager.encoder_ap.sample_rate)
    target_wav = speaker_manager.encoder_ap.load_wav(target_speaker_path, sr=speaker_manager.encoder_ap.sample_rate)
    sys.stdout = sys.__stdout__
    source_wav = torch.from_numpy(source_wav).cuda().unsqueeze(0)
    target_wav = torch.from_numpy(target_wav).cuda().unsqueeze(0)
    coqui_source_emb = speaker_manager.encoder.compute_embedding(source_wav)
    coqui_target_emb = speaker_manager.encoder.compute_embedding(target_wav)
    global COQUI_ENCODER_MODEL 
    COQUI_ENCODER_MODEL = speaker_manager
    return coqui_source_emb, coqui_target_emb

def tortoise_embed(source_speaker_path, target_speaker_path):
    tts = TextToSpeech()
    source_wav = load_voice_path(source_speaker_path)
    target_wav = load_voice_path(target_speaker_path)
    
    tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, _, _ = get_conditioning_latents_torch(tts, source_wav, return_mels=True)
    tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion, _, _ = get_conditioning_latents_torch(tts, target_wav, return_mels=True)
    
    if TORTOISE_AUTOREGRESSIVE_LOSS:
        global TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE 
        TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE = ConditioningEncoder(80, 1024, num_attn_heads=8).cuda()

    if TORTOISE_DIFFUSION_LOSS:
        model_channels = 1024
        in_channels = 100
        num_heads = 16
        global TORTOISE_ENCODER_MODEL_DIFFUSION 
        TORTOISE_ENCODER_MODEL_DIFFUSION = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                        nn.Conv1d(model_channels, model_channels*2,3,padding=1,stride=2),
                                                        AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False)).cuda()
   
    return tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion
        

if __name__ == "__main__":
    
    source_speaker_path = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    
    # Setup RTVC encoders to load source speaker
    print("Loading source speaker...")
    ensure_default_models(Path(RTVC_DEFAULT_MODEL_PATH))
    encoder.load_model(Path(RTVC_DEFAULT_MODEL_PATH + '/default/encoder.pt'))
    
    in_fpath = Path(source_speaker_path.replace("\"", "").replace("\'", ""))
    preprocessed_wav = encoder.preprocess_wav(in_fpath, SAMPLING_RATE)
    wav, _, mel_slices, RTVC_ENCODER_MODEL, _ = encoder.embed_utterance_preprocess(preprocessed_wav, using_partials=True)
    
    wav_tensor_initial = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
    wav_tensor_initial.requires_grad = True

    # Randomly select 10 audio from speaker database
    print("Randomly selecting target speakers...")
    target_speakers_files = []
    for root, dirs, files in os.walk(TARGET_SPEAKER_DATABASE):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                target_speakers_files.append(file_path)
    random.shuffle(target_speakers_files)
    target_speakers_selected = target_speakers_files[:NUM_RANDOM_TARGET_SPEAKER]
    
    # User listens to source and targets, assign score to each
    pygame.mixer.init()
    user_scores = []
    for path in target_speakers_selected:
        print(f"\nSource speaker: {source_speaker_path}")
        print(f"Target speaker: {path}")
        input(f"Press ENTER to listen to the source/target speaker sample pair, then input the difference score...\n")
        
        # Load and play the wav file
        pygame.mixer.music.load(source_speaker_path)
        pygame.mixer.music.play()
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        while True:
            # Prompt user for a score
            score = input(f"Please rate 1-5 how different the target speaker is, 1 being the most similar and 5 being the most different...\n")
            if score in ['1', '2', '3', '4', '5']:  # Check if input is valid
                user_scores.append(int(score))
                break
            else:
                print("Invalid input. Please enter a score between 1-5.\n")

    # Compute source and target embedding differences, also load each encoder model to the global variables
    print("Computing target speakers embedding differences...")
    rtvc_embedding_diffs = []
    avc_embedding_diffs = []
    coqui_embedding_diffs = []
    tortoise_autoregressive_embedding_diffs = []
    tortoise_diffusion_embedding_diffs = []
    
    for target_speaker_path in target_speakers_selected:
        if RTVC_LOSS:
            rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list, rtvc_embed_initial, rtvc_embed_target = rtvc_embed(wav_tensor_initial, mel_slices, target_speaker_path)
            rtvc_embedding_diffs.append(torch.sum(torch.abs(rtvc_embed_initial - rtvc_embed_target)).item())
        if AVC_LOSS:
            avc_embed_initial, avc_embed_target = avc_embed(source_speaker_path, target_speaker_path)
            avc_embedding_diffs.append(torch.sum(torch.abs(avc_embed_initial - avc_embed_target)).item())
        if COQUI_LOSS:
            coqui_embed_initial, coqui_embed_target = coqui_embed(source_speaker_path, target_speaker_path)
            coqui_embedding_diffs.append(torch.sum(torch.abs(coqui_embed_initial - coqui_embed_target)).item())
        if TORTOISE_AUTOREGRESSIVE_LOSS or TORTOISE_DIFFUSION_LOSS:
            tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion = tortoise_embed(source_speaker_path, target_speaker_path)
            if TORTOISE_AUTOREGRESSIVE_LOSS:
                tortoise_autoregressive_embedding_diffs.append(torch.abs(tortoise_source_emb_autoregressive - tortoise_target_emb_autoregressive).sum().item())
            if TORTOISE_DIFFUSION_LOSS:
                tortoise_diffusion_embedding_diffs.append(torch.abs(tortoise_source_emb_diffusion - tortoise_target_emb_diffusion).sum().item())
                
    # Normalize embedding diffs, summing the normalized embedding diffs
    all_lists = [rtvc_embedding_diffs, avc_embedding_diffs, coqui_embedding_diffs, tortoise_autoregressive_embedding_diffs, tortoise_diffusion_embedding_diffs]
    all_lists = [[i / (sum(diffs) / len(diffs)) if diffs else 0 for i in diffs] or [0] * NUM_RANDOM_TARGET_SPEAKER for diffs in all_lists]
    rtvc_embedding_diffs, avc_embedding_diffs, coqui_embedding_diffs, tortoise_autoregressive_embedding_diffs, tortoise_diffusion_embedding_diffs = all_lists
    total_embedding_diffs = [sum(values) for values in zip(*all_lists)]
    
    # Select target speaker that has the largest difference from the source with the analytic hierarchy process
    # Normalize the scores from list1 and list2
    user_scores_weights = np.array(user_scores) / np.sum(user_scores)
    ltotal_embedding_diffs_weights = np.array(total_embedding_diffs) / np.sum(total_embedding_diffs)
    # Aggregate the weights
    overall_weights = 0.5 * user_scores_weights + 0.5 * ltotal_embedding_diffs_weights
    # Find the item with the highest score
    selected_target_speaker_path = target_speakers_selected[np.argmax(overall_weights)]
    
    print('Target selected, preparing attack...')
    # Get selected target speaker's emebdding, preparing the attack
    avc_embed_initial = None
    avc_embed_target = None
    avc_embed_threshold = None
    coqui_embed_initial = None
    coqui_embed_target = None
    coqui_embed_threshold = None
    tortoise_source_emb_autoregressive = None
    tortoise_target_emb_autoregressive = None
    tortoise_threshold_autoregressive = None
    tortoise_source_emb_diffusion = None
    tortoise_target_emb_diffusion = None
    tortoise_threshold_diffusion = None
    rtvc_mel_slices = None
    rtvc_embeds_list = None
    rtvc_frame_tensor_list = None
    rtvc_embed_initial = None
    rtvc_embed_target = None
    rtvc_embed_threshold = None
            
    if RTVC_LOSS:
        rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list, rtvc_embed_initial, rtvc_embed_target = rtvc_embed(wav_tensor_initial, mel_slices, selected_target_speaker_path)
        rtvc_embed_threshold = torch.norm(rtvc_embed_target - rtvc_embed_initial, p=2) * 5
    if AVC_LOSS:
        avc_embed_initial, avc_embed_target = avc_embed(source_speaker_path, selected_target_speaker_path)
        avc_embed_threshold = torch.norm(avc_embed_target - avc_embed_initial, p=2) * 5
    if COQUI_LOSS:
        coqui_embed_initial, coqui_embed_target = coqui_embed(source_speaker_path, selected_target_speaker_path)
        coqui_embed_threshold = torch.norm(coqui_embed_target - coqui_embed_initial, p=2) * 5
    if TORTOISE_AUTOREGRESSIVE_LOSS or TORTOISE_DIFFUSION_LOSS:
        tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion = tortoise_embed(source_speaker_path, selected_target_speaker_path)
        tortoise_threshold_autoregressive = torch.norm(tortoise_target_emb_autoregressive - tortoise_source_emb_autoregressive, p=2) * 100
        tortoise_threshold_diffusion = torch.norm(tortoise_target_emb_diffusion - tortoise_source_emb_diffusion, p=2) * 100
            
    print('Running optimization to find the optimal perturbations...')
    # Run defense
    attack_iteration([wav_tensor_initial], 
                    avc_embed_initial = avc_embed_initial,
                    avc_embed_target = avc_embed_target,
                    avc_embed_threshold = avc_embed_threshold,
                    coqui_embed_initial = coqui_embed_initial,
                    coqui_embed_target = coqui_embed_target,
                    coqui_embed_threshold = coqui_embed_threshold,
                    tortoise_source_emb_autoregressive = tortoise_source_emb_autoregressive,
                    tortoise_target_emb_autoregressive = tortoise_target_emb_autoregressive,
                    tortoise_threshold_autoregressive = tortoise_threshold_autoregressive,
                    tortoise_source_emb_diffusion = tortoise_source_emb_diffusion,
                    tortoise_target_emb_diffusion = tortoise_target_emb_diffusion,
                    tortoise_threshold_diffusion = tortoise_threshold_diffusion,
                    rtvc_mel_slices = rtvc_mel_slices,
                    rtvc_embeds_list = rtvc_embeds_list,
                    rtvc_frame_tensor_list = rtvc_frame_tensor_list,
                    rtvc_embed_initial = rtvc_embed_initial,
                    rtvc_embed_target = rtvc_embed_target,
                    rtvc_embed_threshold = rtvc_embed_threshold,
                    )
    
    print('Source speaker path:' + source_speaker_path)
    print('Target speaker path:' + selected_target_speaker_path)
    print('Output path:' + OUTPUT_DIR)
