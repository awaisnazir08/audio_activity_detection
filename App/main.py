import os
import torch
from .model_class.sound_classifier import LSTMSoundClassifier
import librosa
from .utils.helper_utils import video_to_audio, extract_timestamps, load_config
import numpy as np

def main():
    
    config_file = load_config('App/config.yaml')

    model = LSTMSoundClassifier(input_size=config_file['N_MELS'], hidden_size=config_file['HIDDEN_SIZE'], num_layers=2, output_size=1)
    model.load_state_dict(torch.load('App/model_file/lstm_model.pth', weights_only=True))
    
    # specify the path where the test video is to be saved/placed
    root_path = config_file['VIDEO_DIRECTORY'] 
    video_name = os.listdir(root_path)[0]
    video_path = os.path.join(root_path, video_name)

    # extract audio from video
    video_to_audio(video_path, f'audio_{video_name[5: -4]}.mp3')

    y, sample_rate = librosa.load(f'audio_{video_name[5: -4]}.mp3', sr=config_file['SAMPLE_RATE'], mono=True)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=config_file['SAMPLE_RATE'], n_mels=config_file['N_MELS'], hop_length=config_file['HOP_LENGTH'])
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    mel_spec = mel_spec.unsqueeze(0)
    with torch.no_grad():
        predictions = model(mel_spec).squeeze().cpu().numpy()
    timestamps = extract_timestamps(predictions, hop_length=config_file['HOP_LENGTH'], sr=config_file['SAMPLE_RATE'], merge_threshold=config_file['MERGE_THRESHOLD'], duration_threshold=config_file['DURATION_THRESHOLD'])
    print(timestamps)

if __name__ == "__main__": 
    main()


