# from moviepy.editor import VideoFileClip
from audio_extract import extract_audio
import os
import torch
from model.sound_classifier import LSTMSoundClassifier
import librosa
from utils import video_to_audio, extract_timestamps
import numpy as np

def main():
    model = LSTMSoundClassifier(input_size=64, hidden_size=128, num_layers=2, output_size=1)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()

    root_path = 'test_video'
    video_name = os.listdir(root_path)[0]
    video_path = os.path.join(root_path, video_name)

    video_to_audio(video_path, f'audio_{video_name[5: -4]}.mp3')


    y, sample_rate = librosa.load(f'audio_{video_name[5: -4]}.mp3', sr=22050, mono=True)
    hop_length = 512
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64, hop_length=hop_length)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    mel_spec = mel_spec.unsqueeze(0)
    with torch.no_grad():
        predictions = model(mel_spec).squeeze().cpu().numpy()
        # print('predictions: ', predictions.shape)
    timestamps = extract_timestamps(predictions, hop_length=hop_length, sr=sample_rate, merge_threshold=3)
    print(timestamps)


if __name__ == "__main__": 
    main()


