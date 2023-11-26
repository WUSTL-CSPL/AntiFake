from TTS.api import TTS
import torch
import numpy as np


def compute_embedding_from_clip_torch(speaker_manager, wav_file: str):

    waveform = speaker_manager.encoder_ap.load_wav(wav_file, sr=speaker_manager.encoder_ap.sample_rate)
    
    m_input = torch.from_numpy(waveform)

    # m_input.requires_grad = True
    
    m_input = m_input.cuda()
    
    m_input_0 = m_input.unsqueeze(0)
    
    print('m_input_0 input here')
    print(m_input_0.shape)
    
    m_input_0.requires_grad = True
    
    embedding = speaker_manager.encoder.compute_embedding(m_input_0)
    
    return embedding, m_input_0



if __name__ == "__main__":
    speaker_wav = "voice_samples/p299-reference.wav"
    output_path = "voice_samples/output.wav"

    text = 'This is voice cloning.'
    languages = ['en', 'fr-fr', 'pt-br']

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
              progress_bar=True, gpu=True)
    
    # encoder model is ResNetSpeakerEncoder(BaseEncoder)
    speaker_manager = tts.synthesizer.tts_model.speaker_manager
    speaker_embedding, m_input_0 = compute_embedding_from_clip_torch(speaker_manager, speaker_wav)
    
    # speaker_embedding = tts.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

    # print(speaker_embedding)
    # print(type(speaker_embedding))
    print(speaker_embedding.shape)
    print(m_input_0.shape)
    # print(m_input.requires_grad)


    ## test autoregresive backward to get grad
    loss = speaker_embedding.sum()
    loss.backward()
    grad = m_input_0.grad
    print(grad.shape)
    np.savetxt('autoregressive_grad.txt', grad.cpu().numpy())