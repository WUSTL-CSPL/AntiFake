# AntiFake - Adversarial Perturbation to Protect Unauthorized Speech Synthesis

This repository hosts the source code for the paper "AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech Synthesis". The paper has been accepted by [The 30th ACM Conference on Computer and Communications Security (CCS), 2023](https://www.sigsac.org/ccs/CCS2023/).

AntiFake protects unauthorized speech synthesis of an arbitrary audio sample from attackers by adding adversarial perturbation. AntiFake selects an optimal target speaker and optimizes the source audio sample towards it. The target selection process is a combination of comparing embeddings from a sample database of speakers, as well as human auditory judgments. 

## Dependencies and Setup
AntiFake is implemented and tested on **Python 3.7**. 
We recommend running AntiFake in a virtual environment (e.g. conda) with at least of 16GB of memory space. 

The dependencies are included and can be directly installed via conda:
```bash
conda env create -f antifake.yml 
```

Alternatively, manually install all dependencies by running the following:
```
conda create --name antifake python=3.7
conda activate antifake
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --no-cache
pip install -r ./TTS/requirements.txt
pip install -r requirements.txt
sudo apt install ffmpeg
```

Due to the file size restrictions, some of the files are moved to the supplementary materials hosted on [Drive](https://wustl.box.com/s/ss3wfa94whfrbjnuqau9k31wu7a73p3k). It contains the following files and folders: `antifake_synthesizer_bundle`, `autoregressive.pth`, `diffusion_decoder.pth`, `synthesizer.pt`, and `add_random_noise.py`. The `autoregressive.pth` and `diffusion_decoder.pth` are required to run the AntiFake system, please put them under the path of `AntiFake/tortoise/`; similarly, please put `synthesizer.pt` under the path `saved_models/default/`. The folder named `antifake_synthesizer_bundle` contains the speech synthesizers that can be used to validate the efficacy of the protected samples, and their usage is documented in `antifake_synthesizer_bundle/README_synthesizers.md`. The `add_random_noise.py` script is used for examine the efficacy of the optimized perturbations, for details please see **Additional Experiments**.

# Execution Command Explained

In `run.py` under the "options" block, users can change the options of the output directory **OUTPUT_DIR**, as well as which encoders to use in the ensemble in case VRAM is limited. Additionally, the attack is default in target-based mode, user can also turn on/off **THRESHOLD_BASE** to use threshold base instead. For more details, please refer to [our paper](https://zh1yu4nyu.github.io/files/ZhiyuanYu_CCS23_AntiFake.pdf).

To convert the audio file to the acceptable format of AntiFake, please use the following:
```
ffmpeg -i <source_audio_path> -acodec pcm_s16le -ac 1 -ar 16000 -ab 256k output_audio.wav
```

AntiFake takes in two arguments of a source audio file to be protected (.wav), and a protected output audio file (e.g. "./output/completed.wav")
```
python run.py <source_wav_path> <output_wav_path> 
```

Two examples of running the system are: 
```
python run.py "./samples/libri_2/source/source.wav" "./samples/libri_2/protected/protected.wav"

python run.py "./samples/human_1/source/source.wav" "./samples/human_1/protected/protected.wav"
```
As shown in .samples, the source and protected wav files for the two sample speakers are located in ./samples/<speaker>/source, and ./samples/<speaker>/protected. The synthesized results from the voice cloning engine are also included:

*_rtvc.wav: synthesized using Real Time Voice Cloning (also known as SV2TTS)

*_avc.wav: synthesized using Adaptive Voice Conversion 

*_coqui.wav: synthesized using COQUI TTS

*_tortoise.wav: synthesized using Tortoise TTS

# Additional Experiments

Please note that due to the consideration of computational resources, we currently configure AntiFake to use three of the encoders. Different combinations of encoders used in AntiFake can lead to variations in the protection strength. To configure the encoders used in AntiFake, please change individual encoder "options" in **run.py** to "True".

It might appear straightforward to simply add random noises throughout the entire audio to sufficiently deviate the speaker embedding, without the need for perturbation optimization. To validate the efficacy of our strategy, we extracted the perturbations by referencing the original speech audio, and constructed perturbations that share the same distribution. By applying them to the original audio samples, we found them less effective compared to optimized ones. The script for this comparison experiment can be found in `add_random_noise.py`. The command to use it is:
```
python Add_random_noise.py <protected_wav> <source_wav> <output_wav>
```

# Citation

If you find the platform useful, please cite our work with the following reference:
```
@inproceedings{yu2023antifake,
  title={AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech Synthesis},
  author={Yu, Zhiyuan and Zhai, Shixuan and Zhang, Ning},
  booktitle={Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  pages={460--474},
  year={2023}
}
```
