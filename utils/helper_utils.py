from audio_extract import extract_audio
import os

def video_to_audio(video_path, audio_path):
    """
    Convert a video file to an audio file in .wav format.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the output .wav file.
    """
    try:
        extract_audio(input_path=video_path, output_path=audio_path)

    except Exception as e:
        print(f"An error occurred: {e}")


def extract_timestamps(predictions, hop_length, sr, merge_threshold = 2, duration_threshold = 1):
    timestamps = []
    start = None
    for i, pred in enumerate(predictions):
        if pred > 0.5 and start is None:
            start = i
        elif pred <= 0.5 and start is not None:
            end = i
            timestamps.append((start * hop_length / sr, end * hop_length / sr))
            start = None
    if start is not None:
        timestamps.append((start * hop_length / sr, len(predictions) * hop_length / sr))
    
    merged_timestamps = []
    for ts in timestamps:
        if not merged_timestamps or ts[0] - merged_timestamps[-1][1] > merge_threshold:
            merged_timestamps.append(ts)
        else:
            merged_timestamps[-1] = (merged_timestamps[-1][0], ts[1])
    
    for ts in merged_timestamps:
        if ts[1] - ts[0] < duration_threshold:
            merged_timestamps.remove(ts)
    return merged_timestamps